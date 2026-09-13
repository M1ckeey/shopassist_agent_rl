import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import mimetypes
import shutil
import uuid
import json
from datetime import datetime

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(service="indexing")

class IndexingService:
    def __init__(self):
        self.project_dir = settings.GRAPHRAG_PROJECT_DIR
        self.data_dir_name = settings.GRAPHRAG_DATA_DIR
        self.data_dir = os.path.join(self.project_dir, self.data_dir_name)
        

        # 默认配置文件
        self.default_config = 'settings.yaml'
        self.config_mapping = {
            "text/plain": "settings.yaml",
            "text/markdown": "settings.yaml",
            "application/pdf": "settings_pdf.yaml",
            "text/csv": "settings_csv.yaml",
            "application/csv": "settings_csv.yaml",
            "application/vnd.ms-excel": "settings_csv.yaml",
        }
        
    def _get_file_type(self, file_path: str) -> str:
        """获取文件MIME类型"""
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or 'application/octet-stream'
    
    def _get_config_file(self, file_type: str) -> str:
        """根据文件类型获取对应的配置文件"""
        return self.config_mapping.get(file_type, self.default_config)
    
    def _check_existing_index(self, file_path: str, output_dir: str) -> bool:
        """检查文件是否已经建立索引"""
        file_name = Path(file_path).stem
        index_path = os.path.join(output_dir, f"{file_name}_index")
        return os.path.exists(index_path)
    
    def _prepare_user_directories(self, user_id: int) -> tuple:
        """为用户准备输入和输出目录"""
        # 生成用户UUID
        user_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"user_{user_id}"))
        
        # 创建用户输入目录
        user_input_dir = os.path.join(self.data_dir, "input", user_uuid)
        os.makedirs(user_input_dir, exist_ok=True)
        
        # 创建用户输出目录
        user_output_dir = os.path.join(self.data_dir, "output", user_uuid)
        os.makedirs(user_output_dir, exist_ok=True)
        
        return user_input_dir, user_output_dir
    
    def _copy_file_to_input_dir(self, file_path: str, input_dir: str) -> str:
        """将文件复制到用户的输入目录"""
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(input_dir, file_name)
        
        # 复制文件
        shutil.copy2(file_path, dest_path)
        logger.info(f"已将文件复制到输入目录: {dest_path}")
        
        return dest_path

    def _build_lightweight_index(
        self,
        *,
        file_path: str,
        input_file_path: str,
        file_type: str,
        config_file: str,
        is_update: bool,
        user_id: int,
        user_input_dir: str,
        user_output_dir: str,
    ) -> Dict[str, Any]:
        """Store uploaded file metadata when full GraphRAG indexing is not available."""
        os.makedirs(user_output_dir, exist_ok=True)
        manifest_path = os.path.join(user_output_dir, "lightweight_index.jsonl")
        record = {
            "original_file_path": file_path,
            "input_file_path": input_file_path,
            "file_name": os.path.basename(input_file_path),
            "file_type": file_type,
            "config_used": config_file,
            "is_update": is_update,
            "status": "success",
            "mode": "lightweight",
            "user_id": user_id,
            "input_dir": user_input_dir,
            "output_dir": user_output_dir,
            "indexed_at": datetime.utcnow().isoformat() + "Z",
        }

        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        record["manifest_path"] = manifest_path
        record["message"] = (
            "File saved to the local lightweight index. "
            "Full GraphRAG indexing is bypassed on this machine."
        )
        logger.info(f"Lightweight index saved: {manifest_path}")
        return record
    
    async def process_file(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个文件的索引构建"""
        try:
            file_path = file_info['path']
            file_type = self._get_file_type(file_path)
            user_id = file_info.get('user_id', 0)  # 获取用户ID，默认为0
            
            logger.info(f"开始处理文件: {file_path}, 类型: {file_type}, 用户ID: {user_id}")
            
            # 准备用户目录
            user_input_dir, user_output_dir = self._prepare_user_directories(user_id)
            
            # 复制文件到输入目录
            input_file_path = self._copy_file_to_input_dir(file_path, user_input_dir)
            
            # 获取配置文件
            config_file = self._get_config_file(file_type)
            logger.info(f"使用配置文件: {config_file}")
            
            # 检查是否需要增量更新
            is_update = self._check_existing_index(input_file_path, user_output_dir)

            return self._build_lightweight_index(
                file_path=file_path,
                input_file_path=input_file_path,
                file_type=file_type,
                config_file=config_file,
                is_update=is_update,
                user_id=user_id,
                user_input_dir=user_input_dir,
                user_output_dir=user_output_dir,
            )
            
            # 准备配置
            config_path = os.path.join(self.data_dir, config_file)
            if not os.path.exists(config_path):
                logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
                config_path = os.path.join(self.data_dir, self.default_config)
            
            # 设置配置覆盖
            config_overrides = {
                'input.base_dir': user_input_dir,
                'output.base_dir': user_output_dir,
                # 更新文件匹配模式以匹配文件名
                'input.file_pattern': f".*{os.path.basename(input_file_path)}$$"
            }
            
            # 加载配置
            graphrag_config = load_config(
                Path(self.data_dir),
                Path(config_path),
                config_overrides
            )
            
            # 创建进度记录器
            progress_logger = RichProgressLogger(prefix="graphrag-index")
            
            logger.info(f"开始{'增量更新' if is_update else '构建'}索引: {input_file_path}")
            logger.info(f"输入目录: {user_input_dir}")
            logger.info(f"输出目录: {user_output_dir}")
            
            # 执行索引构建
            index_result = await api.build_index(
                config=graphrag_config,
                method=IndexingMethod.Standard,
                is_update_run=is_update,
                memory_profile=False,
                progress_logger=progress_logger
            )
            
            # 处理结果
            result_info = {
                'original_file_path': file_path,
                'input_file_path': input_file_path,
                'file_type': file_type,
                'config_used': config_file,
                'is_update': is_update,
                'status': 'success',
                'user_id': user_id,
                'input_dir': user_input_dir,
                'output_dir': user_output_dir
            }
            
            # 检查是否有错误
            for workflow_result in index_result:
                if workflow_result.errors:
                    result_info['status'] = 'error'
                    result_info['errors'] = workflow_result.errors
                    logger.error(f"索引构建失败: {workflow_result.errors}")
            
            return result_info
            
        except Exception as e:
            logger.error(f"处理文件时发生错误: {str(e)}", exc_info=True)
            return {
                'file_path': file_path,
                'status': 'error',
                'error': str(e)
            }
    
    async def process_directory(self, directory_path: str, user_id: int = 0) -> Dict[str, Any]:
        """处理整个目录的索引构建"""
        try:
            results = []
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_info = {
                        'path': file_path,
                        'original_name': file,
                        'user_id': user_id
                    }
                    result = await self.process_file(file_info)
                    results.append(result)
            
            return {
                'status': 'success',
                'processed_files': len(results),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"处理目录时发生错误: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e)
            }
