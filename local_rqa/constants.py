# all kinds of constants
from enum import IntEnum, Enum
import os


##### For main modules
OPENAI_MODEL_NAMES = ['gpt-4-1106-preview', 'gpt-4', 'gpt-3.5-turbo', 'text-embedding-ada-002']

class AccelerationFramework(Enum):
    VLLM = 'vllm::'
    TGI = 'tgi::'
    SGLANG = 'sglang::'

##### For the controller and workers (could be overwritten through ENV variables.)
SERVER_LOGDIR = "logs"
CONTROLLER_HEART_BEAT_EXPIRATION = int(
    os.getenv("LOCALRQA_CONTROLLER_HEART_BEAT_EXPIRATION", 90)
)
WORKER_HEART_BEAT_INTERVAL = 15
SERVER_ERROR_MSG = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
QA_MODERATION_MSG = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."
QA_ERROR_MSG = "Sorry I encountered an error. Please try again later or contact support."



class ErrorCode(IntEnum):
    """
    https://platform.openai.com/docs/guides/error-codes/api-errors
    """

    VALIDATION_TYPE_ERROR = 40001

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102
    NO_PERMISSION = 40103

    INVALID_MODEL = 40301
    PARAM_OUT_OF_RANGE = 40302
    CONTEXT_OVERFLOW = 40303

    RATE_LIMIT = 42901
    QUOTA_EXCEEDED = 42902
    ENGINE_OVERLOADED = 42903

    INTERNAL_ERROR = 50001
    CUDA_OUT_OF_MEMORY = 50002
    GRADIO_REQUEST_ERROR = 50003
    GRADIO_STREAM_UNKNOWN_ERROR = 50004
    CONTROLLER_NO_WORKER = 50005
    CONTROLLER_WORKER_TIMEOUT = 50006