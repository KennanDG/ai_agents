import json
import os
import boto3
from functools import lru_cache

_secrets_manager = boto3.client("secretsmanager")

@lru_cache(maxsize=32)
def get_secret_json(secret_arn: str) -> dict:
    res = _secrets_manager.get_secret_value(SecretId=secret_arn)

    return json.loads(res["SecretString"])
