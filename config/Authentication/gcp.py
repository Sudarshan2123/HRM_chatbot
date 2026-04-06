from google.oauth2 import service_account
from google.auth import default, exceptions as auth_exceptions
from google.auth.transport.requests import Request
from google.cloud import resourcemanager_v3
from typing import Optional
import json
import logging
import time
import os

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

# Scopes required for Vertex AI
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def _add_scopes(credentials):
    """Add required scopes to credentials and refresh the token."""
    if hasattr(credentials, "with_scopes"):
        # Service account credentials — add scopes directly
        credentials = credentials.with_scopes(SCOPES)
    elif hasattr(credentials, "scopes") and not credentials.scopes:
        # User credentials without scopes — cannot add scopes, warn the user
        logging.warning(
            "Credentials do not support scopes. "
            "Ensure the OAuth consent screen includes 'https://www.googleapis.com/auth/cloud-platform'."
        )

    # Refresh the token so it's ready to use
    try:
        credentials.refresh(Request())
    except Exception as e:
        logging.warning(f"Token refresh failed (may still work): {e}")

    return credentials


def load_gcp_credentials() -> Optional[service_account.Credentials]:
    """Loads and authenticates GCP credentials with Vertex AI scopes."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Attempt 1: Load default credentials (works in GCP environments)
            credentials, project_id = default(scopes=SCOPES)
            if credentials and project_id:
                try:
                    project_client = resourcemanager_v3.ProjectsClient(credentials=credentials)
                    request = resourcemanager_v3.GetProjectRequest(name=f"projects/{project_id}")
                    project_info = project_client.get_project(request=request)
                    logging.info(
                        f"Attempt {attempt}: GCP default credentials loaded. "
                        f"Project: '{project_info.project_id}'."
                    )
                    return credentials
                except auth_exceptions.GoogleAuthError as auth_error:
                    logging.error(f"Attempt {attempt}: Auth error with default credentials: {auth_error}")
                except Exception as e:
                    logging.error(f"Attempt {attempt}: Error fetching project info: {e}")

                # Return even if project validation failed — credentials may still work
                return credentials

            # Attempt 2: Load from GOOGLE_APPLICATION_CREDENTIALS_CONTENT env var
            if "GOOGLE_APPLICATION_CREDENTIALS_CONTENT" in os.environ:
                try:
                    creds_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_CONTENT"])
                    credentials = service_account.Credentials.from_service_account_info(
                        creds_info, scopes=SCOPES
                    )
                    credentials = _add_scopes(credentials)
                    logging.info(f"Attempt {attempt}: Credentials loaded from GOOGLE_APPLICATION_CREDENTIALS_CONTENT.")
                    return credentials
                except json.JSONDecodeError:
                    logging.error(f"Attempt {attempt}: Error decoding GOOGLE_APPLICATION_CREDENTIALS_CONTENT.")
                except Exception as e:
                    logging.error(f"Attempt {attempt}: Error creating credentials from content: {e}")

            # Attempt 3: Load from GOOGLE_APPLICATION_CREDENTIALS file
            elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                try:
                    credentials = service_account.Credentials.from_service_account_file(
                        os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
                        scopes=SCOPES
                    )
                    credentials = _add_scopes(credentials)
                    logging.info(f"Attempt {attempt}: Credentials loaded from GOOGLE_APPLICATION_CREDENTIALS file.")
                    return credentials
                except FileNotFoundError:
                    logging.error(
                        f"Attempt {attempt}: Service account key file not found: "
                        f"{os.environ['GOOGLE_APPLICATION_CREDENTIALS']}"
                    )
                except Exception as e:
                    logging.error(f"Attempt {attempt}: Error creating credentials from file: {e}")

            else:
                logging.warning(f"Attempt {attempt}: No credentials source found.")

        except Exception as e:
            logging.error(f"Attempt {attempt}: Unexpected error loading GCP credentials: {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    logging.error("Failed to load GCP credentials after all retries.")
    return None