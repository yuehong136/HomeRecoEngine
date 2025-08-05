import logging
import uvicorn

from api.apps import app
from api import settings
from api.db.runtime_config import RuntimeConfig



if __name__ == '__main__':

    logging.info(r"""
============================================================================
                      WELCOME HOME RECOMMENDATION ENGINE!
============================================================================
                """)
    
    # 初始化运行时配置
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default=False, help="Home Recommendation version", action='store_true')
    parser.add_argument('--debug', default=False, help="debug mode", action='store_true')
    args = parser.parse_args()


    RuntimeConfig.DEBUG = args.debug  # 设置调试模式
    if RuntimeConfig.DEBUG:
        logging.info("run on debug mode")

    RuntimeConfig.init_config(JOB_SERVER_HOST=settings.HOST_IP, HTTP_PORT=settings.HOST_PORT)  # 初始化配置 

    uvicorn.run("api.app:app", host=settings.HOST_IP, port=settings.HOST_PORT, log_level="info",
                reload=RuntimeConfig.DEBUG)  # 启动 uvicorn 服务器

