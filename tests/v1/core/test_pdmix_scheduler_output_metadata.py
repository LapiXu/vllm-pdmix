# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.core.sched.output import (
    BatchType,
    HiddenChannelType,
    SchedulerOutput,
)


def test_scheduler_output_has_pdmix_metadata_defaults():
    output = SchedulerOutput.make_empty()

    assert output.batch_type is BatchType.EMPTY
    assert output.head_token is None
    assert output.hidden_channel is None


def test_scheduler_output_accepts_pdmix_metadata_fields():
    output = SchedulerOutput.make_empty()

    output.batch_type = BatchType.PREFILL_FIRST
    output.head_token = "head-1"
    output.hidden_channel = HiddenChannelType.PREFILL_1

    assert output.batch_type is BatchType.PREFILL_FIRST
    assert output.head_token == "head-1"
    assert output.hidden_channel is HiddenChannelType.PREFILL_1
