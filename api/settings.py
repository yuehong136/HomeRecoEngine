import json
import os
import secrets
from datetime import date
from enum import IntEnum, Enum

# import core.utils
# import core.utils.milvus_conn
# import core.utils.infinity_conn, opensearch_conn
from api.constants import HOME_RECOMMENDATION_SERVICE_NAME
from api.utils import get_base_config
from api.utils.file_utils import get_project_base_directory
# from core.nlp import search

LIGHTEN = int(os.environ.get("LIGHTEN", "0"))

LLM = None
LLM_FACTORY = None
LLM_BASE_URL = None
CHAT_MDL = ""
EMBEDDING_MDL = ""
RERANK_MDL = ""
ASR_MDL = ""
IMAGE2TEXT_MDL = ""
API_KEY = None
PARSERS = None
HOST_IP = None
HOST_PORT = None
SECRET_KEY = None
FACTORY_LLM_INFOS = None

# DATABASE_TYPE = os.getenv("DB_TYPE", "postgresql")
# DATABASE = decrypt_database_config(name=DATABASE_TYPE)

# authentication
AUTHENTICATION_CONF = None

# client
CLIENT_AUTHENTICATION = None
HTTP_APP_KEY = None
GITHUB_OAUTH = None
FEISHU_OAUTH = None
OAUTH_CONFIG = None
DOC_ENGINE = None
docStoreConn = None

retrievaler = None
kg_retrievaler = None

# user registration switch
REGISTER_ENABLED = 1


# sandbox-executor-manager
SANDBOX_ENABLED = 0
SANDBOX_HOST = None

BUILTIN_EMBEDDING_MODELS = ["BAAI/bge-large-zh-v1.5@BAAI", "maidalun1020/bce-embedding-base_v1@Youdao"]


def get_or_create_secret_key():
    secret_key = os.environ.get("HOME_RECOMMENDATION_SERVICE_NAME")
    if secret_key and len(secret_key) >= 32:
        return secret_key

    # Check if there's a configured secret key
    configured_key = get_base_config(HOME_RECOMMENDATION_SERVICE_NAME, {}).get("secret_key")
    if configured_key and configured_key != str(date.today()) and len(configured_key) >= 32:
        return configured_key

    # Generate a new secure key and warn about it
    import logging
    new_key = secrets.token_hex(32)
    logging.warning(
        "SECURITY WARNING: Using auto-generated SECRET_KEY. "
        f"Generated key: {new_key}"
    )
    return new_key


def init_settings():
    # global LLM, LLM_FACTORY, LLM_BASE_URL, LIGHTEN, DATABASE_TYPE, DATABASE, FACTORY_LLM_INFOS, REGISTER_ENABLED
    # LIGHTEN = int(os.environ.get('LIGHTEN', "0"))
    # DATABASE_TYPE = os.getenv("DB_TYPE", 'postgresql')
    # DATABASE = decrypt_database_config(name=DATABASE_TYPE)
    # LLM = get_base_config("user_default_llm", {})
    # LLM_FACTORY = LLM.get("factory", "ZHIPU-AI")
    # LLM_BASE_URL = LLM.get("base_url")
    global LLM, LLM_FACTORY, LLM_BASE_URL, LIGHTEN, FACTORY_LLM_INFOS, REGISTER_ENABLED
    LIGHTEN = int(os.environ.get('LIGHTEN', "0"))
    LLM = get_base_config("user_default_llm", {})
    LLM_DEFAULT_MODELS = LLM.get("default_models", {})
    # LLM_FACTORY = LLM.get("factory", "ZHIPU-AI")
    LLM_FACTORY = LLM.get("factory")
    LLM_BASE_URL = LLM.get("base_url")
    try:
        REGISTER_ENABLED = int(os.environ.get("REGISTER_ENABLED", "1"))
    except Exception:
        pass

    try:
        with open(os.path.join(get_project_base_directory(), "conf", "llm_factories.json"), "r") as f:
            FACTORY_LLM_INFOS = json.load(f)["factory_llm_infos"]
    except Exception:
        FACTORY_LLM_INFOS = []

    global CHAT_MDL, EMBEDDING_MDL, RERANK_MDL, ASR_MDL, IMAGE2TEXT_MDL
    if not LIGHTEN:
        EMBEDDING_MDL = BUILTIN_EMBEDDING_MODELS[0]

    if LLM_DEFAULT_MODELS:
        CHAT_MDL = LLM_DEFAULT_MODELS.get("chat_model", CHAT_MDL)
        EMBEDDING_MDL = LLM_DEFAULT_MODELS.get("embedding_model", EMBEDDING_MDL)
        RERANK_MDL = LLM_DEFAULT_MODELS.get("rerank_model", RERANK_MDL)
        ASR_MDL = LLM_DEFAULT_MODELS.get("asr_model", ASR_MDL)
        IMAGE2TEXT_MDL = LLM_DEFAULT_MODELS.get("image2text_model", IMAGE2TEXT_MDL)

        # factory can be specified in the config name with "@". LLM_FACTORY will be used if not specified
        CHAT_MDL = CHAT_MDL + (f"@{LLM_FACTORY}" if "@" not in CHAT_MDL and CHAT_MDL != "" else "")
        EMBEDDING_MDL = EMBEDDING_MDL + (f"@{LLM_FACTORY}" if "@" not in EMBEDDING_MDL and EMBEDDING_MDL != "" else "")
        RERANK_MDL = RERANK_MDL + (f"@{LLM_FACTORY}" if "@" not in RERANK_MDL and RERANK_MDL != "" else "")
        ASR_MDL = ASR_MDL + (f"@{LLM_FACTORY}" if "@" not in ASR_MDL and ASR_MDL != "" else "")
        IMAGE2TEXT_MDL = IMAGE2TEXT_MDL + (f"@{LLM_FACTORY}" if "@" not in IMAGE2TEXT_MDL and IMAGE2TEXT_MDL != "" else "")

    global API_KEY, PARSERS, HOST_IP, HOST_PORT, SECRET_KEY
    API_KEY = LLM.get("api_key", "")
    PARSERS = LLM.get(
        "parsers", "naive:General,qa:Q&A,resume:Resume,manual:Manual,table:Table,paper:Paper,book:Book,laws:Laws,presentation:Presentation,picture:Picture,one:One,audio:Audio,email:Email,tag:Tag"
    )

    HOST_IP = get_base_config(HOME_RECOMMENDATION_SERVICE_NAME, {}).get("host", "127.0.0.1")
    HOST_PORT = get_base_config(HOME_RECOMMENDATION_SERVICE_NAME, {}).get("http_port")

    # SECRET_KEY = get_or_create_secret_key()
    SECRET_KEY = get_base_config(HOME_RECOMMENDATION_SERVICE_NAME, {}).get("secret_key", str(date.today()))

    global AUTHENTICATION_CONF, CLIENT_AUTHENTICATION, HTTP_APP_KEY, GITHUB_OAUTH, FEISHU_OAUTH, OAUTH_CONFIG
    # authentication
    AUTHENTICATION_CONF = get_base_config("authentication", {})

    # client
    CLIENT_AUTHENTICATION = AUTHENTICATION_CONF.get("client", {}).get("switch", False)
    HTTP_APP_KEY = AUTHENTICATION_CONF.get("client", {}).get("http_app_key")
    GITHUB_OAUTH = get_base_config("oauth", {}).get("github")
    FEISHU_OAUTH = get_base_config("oauth", {}).get("feishu")

    OAUTH_CONFIG = get_base_config("oauth", {})

    global DOC_ENGINE, docStoreConn, retrievaler, kg_retrievaler
    DOC_ENGINE = os.environ.get("DOC_ENGINE", "milvus")
    # DOC_ENGINE = os.environ.get('DOC_ENGINE', "opensearch")
    lower_case_doc_engine = DOC_ENGINE.lower()
    # if lower_case_doc_engine == "milvus":
    #     docStoreConn = core.utils.milvus_conn.MilvusConnection()
    # elif lower_case_doc_engine == "opensearch":
    #     docStoreConn = core.utils.opensearch_conn.OSConnection()
    # elif lower_case_doc_engine == "infinity":
    #     docStoreConn = core.utils.infinity_conn.InfinityConnection()
    # else:
    #     raise Exception(f"Not supported doc engine: {DOC_ENGINE}")

    # retrievaler = search.Dealer(docStoreConn)
    # from graphrag import search as kg_search
    # kg_retrievaler = kg_search.KGSearch(docStoreConn)

    # if int(os.environ.get("SANDBOX_ENABLED", "0")):
    #     global SANDBOX_HOST
    #     SANDBOX_HOST = os.environ.get("SANDBOX_HOST", "sandbox-executor-manager")


class CustomEnum(Enum):
    @classmethod
    def valid(cls, value):
        try:
            cls(value)
            return True
        except BaseException:
            return False

    @classmethod
    def values(cls):
        return [member.value for member in cls.__members__.values()]

    @classmethod
    def names(cls):
        return [member.name for member in cls.__members__.values()]


class RetCode(IntEnum, CustomEnum):
    SUCCESS = 0
    NOT_EFFECTIVE = 10
    EXCEPTION_ERROR = 100
    ARGUMENT_ERROR = 101
    DATA_ERROR = 102
    OPERATING_ERROR = 103
    CONNECTION_ERROR = 105
    RUNNING = 106
    PERMISSION_ERROR = 108
    AUTHENTICATION_ERROR = 109
    UNAUTHORIZED = 401
    SERVER_ERROR = 500
    FORBIDDEN = 403
    NOT_FOUND = 404


# AIFORBI
AIFORBI_BASE_CONFIG = get_base_config("aiforbi", {})
AIFORBI_BASE_URL = AIFORBI_BASE_CONFIG.get("base_url")
AIFORBI_API_KEY = AIFORBI_BASE_CONFIG.get("api_key")
AIFORBI_MODEL_ID = AIFORBI_BASE_CONFIG.get("model_id")

AI_TRANSLATE_BASE_CONFIG = get_base_config("ai_translate", {})
AI_TRANSLATE_BASE_URL = AI_TRANSLATE_BASE_CONFIG.get("base_url")
AI_TRANSLATE_API_KEY = AI_TRANSLATE_BASE_CONFIG.get("api_key")
AI_TRANSLATE_MODEL_ID = AI_TRANSLATE_BASE_CONFIG.get("model_id")

SCRIPT_SCHEDULER_BASE_CONFIG = get_base_config("script_scheduler", {})
SCRIPT_SCHEDULER_HOST = SCRIPT_SCHEDULER_BASE_CONFIG.get("host")
SCRIPT_SCHEDULER_PORT = SCRIPT_SCHEDULER_BASE_CONFIG.get("port")

DCS_SERVER_BASE_CONFIG = get_base_config("dcs_server", {})
DCS_SERVER_PROTOCOL = DCS_SERVER_BASE_CONFIG.get("protocol")
DCS_SERVER_HOST = DCS_SERVER_BASE_CONFIG.get("host")
DCS_SERVER_PORT = DCS_SERVER_BASE_CONFIG.get("port")