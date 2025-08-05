import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_login import LoginManager
from fastapi_login.exceptions import InvalidCredentialsException
from fastapi.security import OAuth2PasswordRequestForm
# 添加Session中间件用于OAuth状态管理
# from starlette.middleware.sessions import SessionMiddleware
# from sqlalchemy.orm import Session

# from api.db.db_models import get_db, SessionLocal
# from api.db.services import UserService
from api import settings
from api.utils import get_uuid, current_timestamp, datetime_format
from datetime import datetime
from api.constants import API_VERSION

description = """
Home Recommendation API helps you do awesome stuff. 🚀
"""

# 在模块顶部（路由定义之前）创建线程池
executor = ThreadPoolExecutor(max_workers=20)  # 可以根据服务器性能调整


# 创建FastAPI实例
app = FastAPI(
    title="Home Recommendation",
    description=description,
    summary="AI推荐引擎",
    version="0.1.0",
    terms_of_service="",
    contact={
        "name": "DuXiaolong",
        "url": "https://github.com/yuehong136?tab=repositories",
        "email": "du13013901711@163.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)
# 添加处理CORS（跨域资源共享）的中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源的请求
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)
settings.init_settings()

# 初始化登录管理器，设置密钥和令牌URL
manager = LoginManager(settings.SECRET_KEY, token_url='/auth/token', default_expiry=timedelta(days=1))


# 定义一个函数，用于搜索API和应用页面的路径
def search_pages_path(pages_dir):
    app_path_list = [path for path in pages_dir.glob('*_app.py') if not path.name.startswith('.')]
    api_path_list = [path for path in pages_dir.glob('*_api.py') if not path.name.startswith('.')]
    app_path_list.extend(api_path_list)
    return app_path_list


# 定义一个函数，用于注册页面模块到FastAPI应用中
def register_page(page_path):
    path = f'{page_path}'
    page_name = page_path.stem.removesuffix('_api') if "_api" in path else page_path.stem.removesuffix('_app')
    module_name = '.'.join(page_path.parts[page_path.parts.index('api'):-1] + (page_name,))

    spec = spec_from_file_location(module_name, page_path)
    page = module_from_spec(spec)
    sys.modules[module_name] = page
    spec.loader.exec_module(page)
    sdk_path = "\\sdk\\" if sys.platform.startswith("win") else "/sdk/"
    url_prefix = (
        f"/api/{API_VERSION}" if sdk_path in path else f"/{API_VERSION}/{page_name}"
    )
    # 确保模块有 router 属性
    if hasattr(page, 'router'):
        app.include_router(page.router, prefix=url_prefix, tags=[page_name])
    else:
        logging.warning(f"Module {module_name} does not have 'router' attribute.")


# 定义要搜索页面的目录
pages_dir = [
    Path(__file__).parent,
    Path(__file__).parent.parent / 'api' / 'apps',
]

# 遍历页面目录，注册每个找到的页面
for dir in pages_dir:
    for path in search_pages_path(dir):
        register_page(path)



# 定义一个简单的根端点，返回一条消息
@app.get("/", summary="根目录")
async def root():
    return {"message": "进入docs接口调试文档,请在地址后加/docs,需要标准doc文档,请在地址后加/redoc"}
