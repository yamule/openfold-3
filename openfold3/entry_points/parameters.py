# Copyright 2025 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pathlib import Path

from func_timeout import FunctionTimedOut, func_timeout

from openfold3.core.utils.s3 import download_s3_file

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path("~/.openfold3/").expanduser()
CHECKPOINT_ROOT_FILENAME = "ckpt_root"

OPENFOLD_BUCKET = "openfold"

OPENFOLD_MODEL_CHECKPOINT_REGISTRY = {
    "openfold3_p2_v1": "of3-p2-v1.pt",
}
DEFAULT_CHECKPOINT_NAME = "openfold3_p2_v1"


def download_model_parameters(
    download_dir: Path,
    parameter_name: str,
    force_download: bool = False,
    skip_confirmation: bool = False,
) -> None:
    """Download OpenFold3 model parameters from S3 if not already present.

    Args:
        download_dir: Directory to download the checkpoint file into.
            The file will be saved as ``download_dir / parameters.value.filename``.
        parameters: Which set of parameters to download (e.g. OpenFold3 p2 v1).
        skip_confirmation: If True, skip the interactive yes/no prompt and
            download immediately. Useful when the caller has already obtained
            user consent (e.g. via a higher-level menu).
    """
    download_dir = Path(download_dir)

    checkpoint_file_name = OPENFOLD_MODEL_CHECKPOINT_REGISTRY[parameter_name]
    target_path = download_dir / checkpoint_file_name
    checkpoint_s3_key = f"staging/{checkpoint_file_name}"

    if target_path.exists() and not force_download:
        logger.info("Parameters already present at %s", target_path)
        return

    if not skip_confirmation:
        _TIMEOUT_LEN = 120
        try:
            confirm = func_timeout(
                _TIMEOUT_LEN,
                input,
                args=[
                    f"Download {checkpoint_s3_key} from s3://{OPENFOLD_BUCKET} "
                    f"to {target_path}? (yes/no): "
                ],
            )
        except FunctionTimedOut as timeout_error:
            raise TimeoutError(
                f"No input received within timeout of {_TIMEOUT_LEN}. "
                "Download cancelled. Consider using `setup_openfold` "
                "for initial setup of model parameters."
            ) from timeout_error

        if confirm.lower() not in ["yes", "y"]:
            logger.warning("Download cancelled")
            return

    download_s3_file(OPENFOLD_BUCKET, checkpoint_s3_key, target_path)
