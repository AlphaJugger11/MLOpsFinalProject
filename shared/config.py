# shared/config.py
import os
from typing import Optional


class Config:
    """Configuration parser for environment variables with sensible defaults"""
    
    # Federated Learning Configuration
    FL_ROUNDS: int = int(os.getenv('FL_ROUNDS', '3'))
    IN_DIM: int = int(os.getenv('IN_DIM', '561'))
    NUM_CLASSES: int = int(os.getenv('NUM_CLASSES', '6'))
    FL_PORT: int = int(os.getenv('FL_PORT', '8080'))
    
    # S3 Configuration
    S3_BUCKET: str = os.getenv('S3_BUCKET', 'mlopsprojbucket')
    AWS_REGION: str = os.getenv('AWS_REGION', 'us-east-1')
    USE_S3: bool = os.getenv('USE_S3', 'true').lower() == 'true'
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: Optional[str] = os.getenv('MLFLOW_TRACKING_URI')
    MLFLOW_EXPERIMENT: str = os.getenv('MLFLOW_EXPERIMENT', 'FL_Experiment')
    
    # Client Configuration
    CLIENT_ID: str = os.getenv('CLIENT_ID', '1')
    DATA_PATH: str = os.getenv('DATA_PATH', '/app/federated_clients/client_1.csv')
    FL_SERVER: str = os.getenv('FL_SERVER', 'server:8080')
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validates configuration values.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if cls.FL_ROUNDS < 1:
            raise ValueError(f"FL_ROUNDS must be >= 1, got {cls.FL_ROUNDS}")
        
        if cls.IN_DIM < 1:
            raise ValueError(f"IN_DIM must be >= 1, got {cls.IN_DIM}")
        
        if cls.NUM_CLASSES < 2:
            raise ValueError(f"NUM_CLASSES must be >= 2, got {cls.NUM_CLASSES}")
        
        if cls.FL_PORT < 1 or cls.FL_PORT > 65535:
            raise ValueError(f"FL_PORT must be between 1-65535, got {cls.FL_PORT}")
        
        if not cls.S3_BUCKET:
            raise ValueError("S3_BUCKET cannot be empty")
        
        if not cls.AWS_REGION:
            raise ValueError("AWS_REGION cannot be empty")
        
        return True
    
    @classmethod
    def get_config_dict(cls) -> dict:
        """
        Returns configuration as a dictionary.
        
        Returns:
            Dictionary of configuration values
        """
        return {
            'FL_ROUNDS': cls.FL_ROUNDS,
            'IN_DIM': cls.IN_DIM,
            'NUM_CLASSES': cls.NUM_CLASSES,
            'FL_PORT': cls.FL_PORT,
            'S3_BUCKET': cls.S3_BUCKET,
            'AWS_REGION': cls.AWS_REGION,
            'USE_S3': cls.USE_S3,
            'MLFLOW_TRACKING_URI': cls.MLFLOW_TRACKING_URI,
            'MLFLOW_EXPERIMENT': cls.MLFLOW_EXPERIMENT,
            'CLIENT_ID': cls.CLIENT_ID,
            'DATA_PATH': cls.DATA_PATH,
            'FL_SERVER': cls.FL_SERVER,
        }
    
    @classmethod
    def print_config(cls):
        """Prints current configuration"""
        print("[config] Current configuration:")
        for key, value in cls.get_config_dict().items():
            # Mask sensitive values
            if 'KEY' in key or 'SECRET' in key or 'PASSWORD' in key:
                value = '***' if value else None
            print(f"  {key}: {value}")


# Validate configuration on import
try:
    Config.validate()
except ValueError as e:
    print(f"[config] Warning: Configuration validation failed: {e}")
