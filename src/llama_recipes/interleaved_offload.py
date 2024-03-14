# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import bisect
from collections import OrderedDict
from dataclasses import dataclass
import gc
import weakref

import torch
import torch.nn as nn
from torch.multiprocessing.reductions import StorageWeakRef
from torch._utils import _rebuild_tensor_v2, get_tensor_metadata

from torch.distributed.optim import _apply_optimizer_in_backward

# set this to True to get a fully blocking version
BLOCKING = False
# set this to True to get a fully-in-stream-sync version (this is more restrictive than BLOCKING)
SYNC_STREAM = False


# Assumptions (important!):
# 1. model must contain a sequence of the given block_type
#    a. sequence means a nn.ModuleList or nn.Sequential sub-module s.t.
#       during forward, the blocks are called in the order they are registered
#       with to the parent module: both forward and backward passes must follow
#       the order given by how the blocks are registered, see also next point
#    b. blocks can have inputs from any earlier block and provide outputs to
#       to any later block than the immediate neighbor (skip connections)
#       as long as they also contain a direct input tensor from the previous block
#       and output tensor to the next block
# 2. outside of the blocks (and in particular between blocks), the main PyTorch
#    stream must be used for computation, or it must be ensured that this stream
#    waits on any asynchronous computation to be done before going into the next block
# 3. parameters must not be shared between different blocks
# 4. within a block, all operations are supported
# 5. TODO: at some point, this should be extended to ensure we discover order
#    of blocks automatically (similar to static_graph in DDP / FSDP)


def _release_storage(t):
    t.untyped_storage().resize_(0)


def _offload(t, cpu_t, non_blocking=not BLOCKING):
    cpu_t.copy_(t, non_blocking=non_blocking)
    _release_storage(t)


def _prefetch(t, cpu_t, non_blocking=not BLOCKING):
    t.untyped_storage().resize_(cpu_t.untyped_storage().size())
    t.copy_(cpu_t, non_blocking=non_blocking)


def _is_offloaded(t):
    return t.untyped_storage().size() == 0


if BLOCKING:
    def _sync_if_blocking(x):
        x.synchronize()
        torch.cuda.current_stream().synchronize()
else:
    def _sync_if_blocking(x):
        pass


def _main_wait_on_custom_event(event):
    torch.cuda.current_stream().wait_event(event)
    _sync_if_blocking(event)


def _custom_stream_wait_on_event(stream, event):
    stream.wait_event(event)
    _sync_if_blocking(stream)


@dataclass
class OffloadRef:
    block_id: int
    key: int


@dataclass
class MainOffloadMeta:
    meta_args: tuple # args passed to _rebuild_tensor_v2
    dtype: torch.dtype
    device: torch.device # always the original device of tensor
    active_tensor: torch.Tensor # tensor having currently active values (may require sync)
    original_tensor: torch.Tensor # reference to original tensor kept until offloading is complete
    num_views: int=0


@dataclass
class ViewOffloadMeta:
    meta_args: tuple=() # args passed to _rebuild_tensor_v2
    dtype: torch.dtype=None


def _make_view_meta(t: torch.Tensor):
    backward_hooks = OrderedDict()  # we don't support hooks for now
    meta_args = (
        t.storage_offset(),
        tuple(t.size()),
        t.stride(),
        t.requires_grad,
        backward_hooks,
        get_tensor_metadata(t),
    )
    return ViewOffloadMeta(meta_args, t.dtype)


def _make_main_meta(t: torch.Tensor, device: torch.device,
                    active_tensor: torch.Tensor, original_tensor: torch.Tensor):
    view = _make_view_meta(t)
    return MainOffloadMeta(
        view.meta_args, view.dtype, device, active_tensor, original_tensor
    )


def _get_main_entries(block):
    def get_main_entry(key, entry):
        # ignore any previously finalized entries
        if isinstance(entry, OffloadRef):
            return None, None
        # get the actual owning block and final key
        main_ref = block.all_packed[key]
        main_block = block.id_map[main_ref.block_id]
        # main entry is always the first one
        main_entry = main_block.packed_tensors[main_ref.key][0]
        assert isinstance(main_entry, MainOffloadMeta)
        return main_ref.key, main_entry

    main_entries = (get_main_entry(key, entry) for key, entry in block.packed_tensors.items())
    # make a dictionary: ensures that we only get each key once
    return {
        k: v for k, v in main_entries if v is not None and v.original_tensor is not None
    }


def _offload_packed(block, non_blocking=not BLOCKING):
    # since we may be deleting tensors here, that will cause finalized
    # entries to be added, we first get all the entries, then apply offloading
    main_entries = _get_main_entries(block)

    # issue offloading on the custom stream
    with torch.cuda.stream(block.act_stream):
        for main_entry in main_entries.values():
            main_entry.active_tensor.copy_(main_entry.original_tensor, non_blocking=non_blocking)
    # record an event which ensures that activations for this block have
    # been packed and offloaded
    block.act_stream.record_event(block.act_event_off)


def _remove_original_tensors(block):
    main_entries = _get_main_entries(block)
    for main_entry in main_entries.values():
        main_entry.original_tensor = None


def _preload_packed(block, non_blocking=not BLOCKING):
    with torch.cuda.stream(block.act_stream):
        for key, entry in block.packed_tensors.items():
            # first, we ignore entries that are simply references to a finalized entry
            if isinstance(entry, OffloadRef):
                continue
            # get the actual owning block and final key
            main_ref = block.all_packed[key]
            main_block = block.id_map[main_ref.block_id]
            # main entry is always the first one
            main_entry = main_block.packed_tensors[main_ref.key][0]
            assert isinstance(main_entry, MainOffloadMeta)
            if main_entry.device == main_entry.active_tensor.device:
                # already pre-loaded (e.g. by another block)
                continue

            # we can over-write the reference to original tensor here because
            # this must be called in our custom stream and the custom stream
            # owns the active_tensor (no matter on which device)
            t = torch.empty_like(main_entry.active_tensor, device=main_entry.device)
            t.copy_(main_entry.active_tensor, non_blocking=non_blocking)
            main_entry.active_tensor = t
    # register an event for the completion of pre-loading
    block.act_stream.record_event(block.act_event_off)


def _cleanup_packed(block, log_domain):
    # clean-up any packed tensors
    if block.is_first:
        block.all_packed.clear()
    if block.packed_tensors:
        # check if we had any remaining main entries
        main_entry = next((
            v[0] for k, v in block.packed_tensors.items()
            if isinstance(v, list) and isinstance(v[0], MainOffloadMeta)
        ), None)
        if main_entry:
            print(
                f"Warning: {log_domain} block {block.block_id}: found "
                f"main entry in packed tensors: {main_entry}"
            )
            _main_wait_on_custom_event(block.act_event_off)
        block.packed_tensors.clear()


class OffloadPreHook(torch.autograd.Function):
    @staticmethod
    def forward(offload_args, *args):
        block, is_grad_tracing = offload_args
        torch.cuda.nvtx.range_push(f"forward {block.block_id}")
        # print(f"block {block.block_id} pre-forward")
        # pre-forward
        if not block._optim_checked and is_grad_tracing:
            block.has_optim_in_backward = any(
                hasattr(p, "_in_backward_optimizers") for p, _ in block.params
            )
            block._optim_checked = True

        if block.is_first:
            # in the very first block, the prefetch stream must wait on the
            # offload events to ensure that we take over ownership of any
            # offloaded parameters
            # for this, we simply wait on both the first and last block,
            # which ensures that whether forward or backward was previously
            # called, prefetch stream now owns the parameters
            _custom_stream_wait_on_event(
                block.prefetch_stream, block.offload_event)
            _custom_stream_wait_on_event(
                block.prefetch_stream,
                block.id_map[block.num_blocks - 1].offload_event)

        # clean-up any packed tensors (post-backward may not be enough because
        # block 0 may not even have any post-backward)
        _cleanup_packed(block, "pre-forward")

        # we should wait on the offloading of previous blocks to ensure that
        # we don't use more memory than necessary (offload stream must not run
        # ahead of main stream)
        prev_block_id = block.block_id - block.num_blocks_params + 1
        if not is_grad_tracing and block.num_blocks_params <= block.num_blocks:
            prev_block_id = prev_block_id % block.num_blocks
        prev_block_p = block.id_map.get(prev_block_id)
        if prev_block_p is not None and block.num_blocks_params > 1:
            _custom_stream_wait_on_event(block.prefetch_stream, prev_block_p.offload_event)

        with torch.cuda.stream(block.prefetch_stream):
            # we prefetch on custom stream: next block will have the main stream
            # wait on this when it gets to pre-forward
            next_block_id = block.block_id + block.num_blocks_params - 1
            if not is_grad_tracing:
                next_block_id = next_block_id % block.num_blocks
            next_block_p = block.id_map.get(next_block_id)
            if next_block_p is not None:
                # only prefetch parameters that were actually offloaded:
                # it's possible we don't offload parameters for optimizer
                for p, cpu_param in next_block_p.params:
                    if _is_offloaded(p):
                        _prefetch(p, cpu_param)
        # mark the prefetching of parameters for next block
        if next_block_p is not None:
            block.prefetch_stream.record_event(next_block_p.prefetch_event)

        # must wait on the prefetching of our parameters to be done
        _main_wait_on_custom_event(block.prefetch_event)

        return args

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.block = inputs[0][0]

    @staticmethod
    def backward(ctx, *grad_args):
        # print(f"block {ctx.block.block_id} post-backward")
        # post-backward

        # record an event signaling that main stream computation of this block
        # is done
        ctx.block.main_event.record()

        # offload-stream must wait on main here to ensure that we can offload
        # the updated parameters
        _custom_stream_wait_on_event(ctx.block.offload_stream, ctx.block.main_event)

        with torch.cuda.stream(ctx.block.offload_stream):
            # offload in custom stream: the custom stream must have seen the
            # effects of updated parameters due to waiting on main stream above
            prev_block_p = ctx.block.id_map.get(
                ctx.block.block_id - ctx.block.num_blocks_params + 1)
            if prev_block_p is not None:
                # only offload here if we have optimizer in backward
                do_offload = prev_block_p.has_optim_in_backward
                for p, cpu_param in ctx.block.params:
                    if do_offload or p.grad is None:
                        _offload(p, cpu_param)
        ctx.block.offload_stream.record_event(ctx.block.offload_event)

        if ctx.block.num_blocks_params == 1:
            # special case: immediately wait on offloading to free up memory
            _main_wait_on_custom_event(ctx.block.offload_event)

        # we let the act stream wait on main here, s.t. prefetching
        # activations does not run ahead of main stream too much (would
        # cause too many memory allocations otherwise)
        _custom_stream_wait_on_event(ctx.block.act_stream, ctx.block.main_event)

        # clean-up any packed tensors (we'll cleanup in pre-forward as well
        # since this may not be enough, but we can already free up the structure)
        # this is done on the main stream which should own any tensors at this point
        # (due to waiting for prefetch stream in pre-backward)
        _cleanup_packed(ctx.block, "post-backward")

        torch.cuda.nvtx.range_pop()
        return None, *grad_args


class OffloadPostHook(torch.autograd.Function):
    @staticmethod
    def forward(offload_args, *args):
        block, is_grad_tracing = offload_args
        # print(f"block {block.block_id} post-forward")
        # post-forward
        # record an event signaling that main stream computation of this block
        # is done
        block.main_event.record()

        if is_grad_tracing:
            # let the act stream wait on main here to ensure that we can
            # offload packed tensors in bulk.
            _custom_stream_wait_on_event(block.act_stream, block.main_event)
            _offload_packed(block)

            # delete the original tensors of the previous block.
            # for this, main has to wait for the offload event, and then delete
            # event of the block previous to that one
            prev_block_act = block.id_map.get(block.block_id - block.num_blocks_act + 1)
            if prev_block_act is not None:
                _main_wait_on_custom_event(prev_block_act.act_event_off)
                # deleting the original tensors can now happen on the main stream,
                # and no additional wait is required
                _remove_original_tensors(prev_block_act)

        # we let the offload stream wait on main stream here to ensure that
        # we can release storage of parameters on that stream
        _custom_stream_wait_on_event(block.offload_stream, block.main_event)

        with torch.cuda.stream(block.offload_stream):
            next_block_p = block.id_map.get(block.block_id + block.num_blocks_params - 1)
            do_offload = (
                (not is_grad_tracing and block.num_blocks_params <= block.num_blocks) or
                next_block_p is not None
            )
            if do_offload:
                for p, cpu_param in block.params:
                    _offload(p, cpu_param)
        # mark the offloading of our parameters
        block.offload_stream.record_event(block.offload_event)

        if block.is_last and is_grad_tracing:
            # in the last block, ensure that the prefetch stream waits on
            # offload stream s.t. it owns any offloaded parameters
            _custom_stream_wait_on_event(block.prefetch_stream, block.offload_event)

        if block.num_blocks_params == 1:
            # special case: should wait on offloaded parameters immediately
            _main_wait_on_custom_event(block.offload_event)

        torch.cuda.nvtx.range_pop()
        return args[0] if len(args) == 1 else args

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.block = inputs[0][0]

    @staticmethod
    def backward(ctx, *grad_args):
        # print(f"block {ctx.block.block_id} pre-backward")
        torch.cuda.nvtx.range_push(f"backward {ctx.block.block_id}")
        # pre-backward

        # we should wait on the offloading of previous blocks to ensure that
        # we don't use more memory than necessary (offload stream must not run
        # ahead of main stream)
        next_block_id = ctx.block.block_id + ctx.block.num_blocks_params - 1
        next_block_p = ctx.block.id_map.get(next_block_id)
        if next_block_p is not None and ctx.block.num_blocks_params > 1:
            _custom_stream_wait_on_event(ctx.block.prefetch_stream, next_block_p.offload_event)

        with torch.cuda.stream(ctx.block.prefetch_stream):
            # we prefetch on custom stream: custom stream waits on main stream
            # during backward, which ensures that parameters have been released
            # prior to this
            prev_block_p = ctx.block.id_map.get(
                ctx.block.block_id - ctx.block.num_blocks_params + 1)
            if prev_block_p is not None:
                for p, cpu_param in prev_block_p.params:
                    _prefetch(p, cpu_param)
        if prev_block_p is not None:
            ctx.block.prefetch_stream.record_event(prev_block_p.prefetch_event)

        _main_wait_on_custom_event(ctx.block.prefetch_event)

        # pre-load packed activations: this must be done entirely in
        # act stream, since it owns the offloaded (and prefetched)
        # tensors. main stream will wait for this pre-loading in the
        # pre-backward of prev_block when called, before any unpacking happens
        prev_block_act = ctx.block.id_map.get(
            ctx.block.block_id - ctx.block.num_blocks_act + 1)
        if prev_block_act is not None:
            _preload_packed(prev_block_act)
        # wait on the pre-loading for this block to be done
        # for the last block, we never pack anything, so avoid the wait
        if not ctx.block.is_last:
            _main_wait_on_custom_event(ctx.block.act_event_off)

        return None, *grad_args


class OffloadBlockWrapper(nn.Module):
    def __init__(
                self,
                block,
                block_id,
                block_id_map,
                all_params,
                all_packed,
                prefetch_stream,
                offload_stream,
                act_stream,
                only_frozen_params
            ):
        super().__init__()
        self.block = block
        self.block_id = block_id
        self.id_map = block_id_map
        self.all_params = all_params
        self.all_packed = all_packed
        self.prefetch_stream = prefetch_stream
        self.offload_stream = offload_stream
        self.act_stream = act_stream
        self.only_frozen_params = only_frozen_params
        self.num_blocks_params = 0
        self.num_blocks_act = 0
        self.num_blocks = 0
        self.packed_tensors = dict()
        self.params = []
        self.is_first = block_id == 0
        self.is_last = None
        self.has_optim_in_backward = False
        self._optim_checked = False

    def initialize(self, num_blocks_params, num_blocks_act, device):
        self.num_blocks_params = num_blocks_params
        self.num_blocks_act = num_blocks_act
        self.num_blocks = len(self.id_map)
        self.is_last = self.block_id == self.num_blocks - 1
        self.prefetch_event = torch.cuda.Event()
        self.offload_event = torch.cuda.Event()
        self.act_event_off = torch.cuda.Event()
        self.main_event = torch.cuda.Event()

        # ensure that we have copies for both parameters and gradients
        # placed on the right device
        def init_device(m):
            for param_key, param in m._parameters.items():
                if param is None:
                    continue
                assert isinstance(param, nn.Parameter)
                assert param.grad is None, "Must init OffloadingWrapper before gradients"
                no_trainable_offload = self.only_frozen_params and param.requires_grad
                # TODO: this way of determining "never offload" only applies to
                # training workloads (in forward-only, we may want to offload every block)
                never_offload = (self.block_id >= self.num_blocks - self.num_blocks_params or
                    no_trainable_offload)
                if never_offload:
                    # just move to device and continue
                    m._parameters[param_key] = nn.Parameter(param.to(device=device), param.requires_grad)
                    continue

                with torch.cuda.stream(self.offload_stream):
                    if self.block_id < self.num_blocks_params:
                        # just move current parameter to the right device
                        new_p = nn.Parameter(param.to(device=device), param.requires_grad)
                        cpu_p = torch.empty_like(param, device="cpu", pin_memory=True)
                    else:
                        # create an empty parameter on device with right dimensions
                        # this causes additional GPU memory, but only for one
                        # param at a time. Otherwise, we could construct a dummy
                        # tensor just for the storage (with new device), then
                        # _rebuild_tensor with the real meta args, and release the storage
                        new_p = nn.Parameter(
                            torch.empty_like(param, device=device), param.requires_grad
                        )
                        _release_storage(new_p)
                        # this may also create additional CPU memory, but only
                        # one parameter at a time
                        cpu_p = param.cpu().pin_memory()

                if not (self.block_id < self.num_blocks_params and
                        self.block_id >= self.num_blocks - self.num_blocks_params):
                    # for blocks that may be offloaded at some point, we register
                    # the keys for the activation packing to check against them
                    key = hash(StorageWeakRef(new_p.untyped_storage()))
                    if key in self.all_params and self.all_params[key] != self.block_id:
                        raise ValueError(
                            "Cannot share parameters between blocks: "
                            f"Storage with hash {key} owned by {self.all_params[key]} "
                            f"but also by {self.block_id}"
                        )
                    self.all_params[key] = self.block_id

                # important: actually replace the parameter in original module
                m._parameters[param_key] = new_p
                self.params.append((new_p, cpu_p))

            # move buffers to requested device always
            for buf_key, buf in m._buffers.items():
                if buf is not None:
                    m._buffers[buf_key] = buf.to(device=device)

        with torch.no_grad():
            self.block.apply(init_device)

        # ensure that CPU parameters have been allocated for main stream
        # before continuing
        self.offload_stream.record_event(self.offload_event)
        _main_wait_on_custom_event(self.offload_event)
        # ensure that parameters are actually released for the custom stream:
        # at this point, the offload stream owns the parameters, as would be
        # the case after a full backward
        self.main_event.record()
        _custom_stream_wait_on_event(self.offload_stream, self.main_event)

        # for activation offloading, we need both pre-backward and post-backward
        # hooks, so we just use the full forward anyway
        if self.num_blocks_params > self.num_blocks and self.num_blocks_act > self.num_blocks:
            self.forward = self._forward_none
        else:
            self.forward = self._forward_full

        # ensure any storage associated with original block is released
        gc.collect()

    def pack(self, t):
        if self.block_id >= self.num_blocks - self.num_blocks_act or not t.is_cuda:
            # we're in the last device blocks, so don't pack anything anymore
            return t

        storage = t.untyped_storage()
        key = hash(StorageWeakRef(storage))
        # hashes must be strictly positive
        assert key > 0

        # if this is actually a parameter, don't pack it
        if key in self.all_params and self.num_blocks_params >= self.num_blocks_act:
            return t

        do_offload = True
        new_key = key
        while new_key in self.all_packed:
            ref = self.all_packed[new_key]
            if ref.key < 0:
                # this has been finalized already, we choose a different key
                new_key += 1
                continue
            # `ref` must point to a "main" storage, thus we don't need to offload
            do_offload = False
            break

        if do_offload:
            assert new_key not in self.packed_tensors
            # cannot offload here immediately because tensor `t` may not be
            # valid at all (at least w.r.t. custom stream). We'll offload
            # in bulk in post-forward
            with torch.no_grad():
                with torch.cuda.stream(self.act_stream):
                    cpu_t = torch.empty_like(t, device="cpu", pin_memory=True)
                    meta = _make_main_meta(t, t.device, cpu_t, t)
            view_idx = 0
            main_meta = meta
            self.packed_tensors[new_key] = [meta]
            self.all_packed[new_key] = OffloadRef(self.block_id, new_key)
        else:
            # this must be a view to some existing (non-finalized) storage
            # we get that reference from `all_packed`, and add the view entry
            # to this block's packed tensors
            meta = _make_view_meta(t)
            main_ref = self.all_packed[new_key]
            # add the view entry
            entries = self.packed_tensors.get(new_key, [])
            view_idx = len(entries)
            entries.append(meta)
            self.packed_tensors[new_key] = entries
            # assign the main meta to increase `num_views` correctly
            block = self.id_map[main_ref.block_id]
            main_meta = block.packed_tensors[main_ref.key][0]

        # whether we are the "main" view or not, we always increase the number
        # of views by exactly 1
        main_meta.num_views += 1

        def on_storage_del(k):
            if k not in self.packed_tensors:
                return
            entries = self.packed_tensors[k]
            if isinstance(entries, OffloadRef):
                # already called this, print warning
                print(
                    f"Warning: finalize called twice for storage with key {k}: {entries}"
                )
                return
            # it's important to assign a new key that can never clash with
            # any hash an actual storage could produce. This is why we use
            # the negative number range here
            new_k = -k
            while new_k in self.all_packed:
                new_k -= 1
            assert new_k not in self.packed_tensors
            new_ref = OffloadRef(self.block_id, new_k)
            self.all_packed[k] = new_ref
            self.all_packed[new_k] = new_ref
            self.packed_tensors[new_k] = entries
            # put the new ref in `packed_tensors` as well: anyone trying to
            # access main storage should go through `all_packed`
            self.packed_tensors[k] = new_ref

        # only need to add a finalize hook in case we are the main owner
        if do_offload:
            weakref.finalize(storage, on_storage_del, new_key)

        return new_key, view_idx

    def unpack(self, key_idx):
        if isinstance(key_idx, torch.Tensor):
            return key_idx

        key, view_idx = key_idx
        # get the actual block and key for this entry
        main_ref = self.all_packed[key]
        main_block = self.id_map[main_ref.block_id]
        main_entry = main_block.packed_tensors[main_ref.key][0]
        assert isinstance(main_entry, MainOffloadMeta)
        try:
            # this tensor has already been moved back to device
            main_t = main_entry.active_tensor
            if main_ref.block_id == self.block_id and view_idx == 0:
                # this is the special case where block ownership did not change
                # and we have the "main" view: can directly return the tensor
                return main_t
            # in all other cases, need to get our view meta and re-construct tensor
            # if the main entry is in another block, the key for our view entries
            # may not have changed, just check this here
            self_entries = self.packed_tensors[key]
            if isinstance(self_entries, OffloadRef):
                view_meta = self.packed_tensors[self_entries.key][view_idx]
            else:
                view_meta = self_entries[view_idx]
            storage = torch.storage.TypedStorage(
                wrap_storage=main_t._typed_storage()._untyped_storage,
                dtype=view_meta.dtype,
                _internal=True,
            )
            return _rebuild_tensor_v2(storage, *view_meta.meta_args)
        finally:
            # removing any meta entry always decreases the `num_views` of the
            # main meta by exactly 1
            main_entry.num_views -= 1
            # the (prefetched) tensor was created by the act stream,
            # so delete it on that stream as well
            if main_entry.num_views == 0:
                with torch.cuda.stream(self.act_stream):
                    del main_block.packed_tensors[main_ref.key]

    def _forward_none(self, *args, **kwargs):
        return self.block(*args, **kwargs)

    def _forward_full(self, *args, **kwargs):
        is_grad_tracing = torch.is_grad_enabled()
        args = OffloadPreHook.apply((self, is_grad_tracing), *args)
        with torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack):
            args = self.block(*args, **kwargs)
        # TODO use actual hooks for the layers instead of this sorcery
        if isinstance(args, torch.Tensor):
            pack, args = False, (args,)
        elif len(args) == 1 and isinstance(args[0], torch.Tensor):
            pack, args = True, args
        else:
            pack, args = False, args
        args = OffloadPostHook.apply((self, is_grad_tracing), *args)
        return (args,) if pack else args


class OffloadingWrapper(nn.Module):
    def __init__(self, wrapped_module, block_type,
                 device=None, num_blocks_params=2, num_blocks_act=2,
                 only_frozen_params=False):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.block_type = block_type
        self.device = device or torch.cuda.current_device()
        if SYNC_STREAM:
            self.prefetch_stream = torch.cuda.current_stream()
            self.offload_stream = self.prefetch_stream
            self.act_stream = self.prefetch_stream
        else:
            self.prefetch_stream = torch.cuda.Stream()
            self.offload_stream = torch.cuda.Stream()
            self.act_stream = torch.cuda.Stream()
        self.block_id_map = OrderedDict()
        self.all_params = dict()
        self.all_packed = dict()
        block_id = 0
        current_stack = [(
            wrapped_module, None, None, isinstance(wrapped_module, block_type)
        )]
        current_item = 0
        while current_item < len(current_stack):
            m, name, parent, within_block = current_stack[current_item]
            is_block = isinstance(m, block_type)
            if is_block:
                if parent is None:
                    raise ValueError(
                        "Block type cannot be type of main wrapped module"
                    )
                new_module = OffloadBlockWrapper(
                    m, block_id, self.block_id_map, self.all_params,
                    self.all_packed, self.prefetch_stream, self.offload_stream,
                    self.act_stream, only_frozen_params
                )
                setattr(parent, name, new_module)
                self.block_id_map[block_id] = new_module
                block_id += 1
            elif not within_block:
                m._apply(lambda t: t.to(device=self.device), recurse=False)
            current_item += 1
            current_stack[current_item:current_item] = [
                (child, name, m, is_block or within_block)
                for name, child in m.named_children()
            ]
        # ensure that we don't keep references to the original blocks anywhere
        del current_stack[:]
        gc.collect()

        num_blocks_params = min(num_blocks_params, len(self.block_id_map) + 1)
        if num_blocks_params <= 0:
            num_blocks_params = len(self.block_id_map) + 1
        num_blocks_act = min(num_blocks_act, len(self.block_id_map) + 1)
        if num_blocks_act <= 0:
            num_blocks_act = len(self.block_id_map) + 1
        for block_wrapper in self.block_id_map.values():
            block_wrapper.initialize(num_blocks_params, num_blocks_act, self.device)

    def forward(self, *args, **kwargs):
        return self.wrapped_module(*args, **kwargs)


class TestBlock(nn.Module):
    def __init__(self, in_size, out_size, act=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.relu = act() if act else None

    def forward(self, x):
        x = self.linear(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def main():
    batch_size, in_size, hidden_size, out_size = 1024, 128, 512, 256
    device = torch.device("cuda")

    model = nn.Sequential(
        TestBlock(in_size, hidden_size),
        TestBlock(hidden_size, hidden_size),
        TestBlock(hidden_size, hidden_size),
        TestBlock(hidden_size, out_size, act=None)
    )
    model = OffloadingWrapper(model, TestBlock, device=device)
    loss = nn.CrossEntropyLoss()

    for optim_in_backward in [True, False]:
        print(f"using optim in backward: {optim_in_backward}")
        if optim_in_backward:
            _apply_optimizer_in_backward(
                torch.optim.AdamW,
                params=model.parameters(),
                optimizer_kwargs={"lr": 2.0}
            )
            optimizer = None
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=2.0)
            optimizer.zero_grad()

        for i in range(8):
            print(f"iteration {i}", end=" ", flush=True)
            # p_wrapped = next(model.wrapped_module.children()).params[0]
            # p_linear = next(model.wrapped_module.children()).block.linear.weight
            # print(p_wrapped[0] is p_linear)
            # print(p_wrapped)
            torch.cuda.nvtx.range_push(f"IT {i}")
            inputs = torch.randn(batch_size, in_size, device=device)
            targets = torch.randint(0, out_size, size=(batch_size,), device=device)
            torch.cuda.nvtx.range_push(f"forward")
            outputs = model(inputs)
            loss_value = loss(outputs, targets)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push(f"backward")
            # import pdb; pdb.set_trace()
            loss_value.backward()
            if optimizer:
                optimizer.step()
                optimizer.zero_grad()
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()
            print(f"done")


if __name__ == "__main__":
    main()
