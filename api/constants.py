NAME_LENGTH_LIMIT = 2**10

IMG_BASE64_PREFIX = "data:image/png;base64,"

SERVICE_CONF = "service_conf.yaml"

API_VERSION = "v1"
HOME_RECOMMENDATION_SERVICE_NAME = "home_recommendation"
REQUEST_WAIT_SEC = 2
REQUEST_MAX_WAIT_SEC = 300

DATASET_NAME_LIMIT = 128
# milvus的集合命名规则：以字母开头，只能包含字母、数字和下划线
MILVUS_NAME_PATTERN = r"^[a-zA-Z][a-zA-Z0-9_]*$"
FILE_NAME_LEN_LIMIT = 255