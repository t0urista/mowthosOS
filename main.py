"""
FastAPI microservice for Mammotion robotic mower control.

This service provides a REST API interface to the PyMammotion library,
allowing remote control of Mammotion robotic mowers.
"""

import asyncio
import logging
import random
import traceback
from typing import Any, Dict, Optional
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from pymammotion.mammotion.devices.mammotion import Mammotion
from pymammotion.aliyun.cloud_gateway import CloudIOTGateway, SetupException
from pymammotion.aliyun.model.dev_by_account_response import Device
from pymammotion.data.model.device import MowingDevice
from pymammotion.data.model.enums import ConnectionPreference
from pymammotion.data.mower_state_manager import MowerStateManager
from pymammotion.http.http import MammotionHTTP
from pymammotion.http.model.camera_stream import StreamSubscriptionResponse, VideoResourceResponse
from pymammotion.http.model.http import Response
from pymammotion.mammotion.devices.mammotion_bluetooth import MammotionBaseBLEDevice
from pymammotion.mammotion.devices.mammotion_cloud import MammotionBaseCloudDevice, MammotionCloud
from pymammotion.mqtt import MammotionMQTT
from pymammotion.utility.device_type import DeviceType
from pymammotion.utility.constant.device_constant import WorkMode, device_mode

TIMEOUT_CLOUD_RESPONSE = 10

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Mammotion Mower Control API",
    description="REST API for controlling Mammotion robotic mowers",
    version="1.0.0"
)

# Pydantic models for request/response
class LoginRequest(BaseModel):
    account: str = Field(..., description="Mammotion account email/username")
    password: str = Field(..., description="Account password")
    device_name: Optional[str] = Field(None, description="Specific device name to connect to")

class LoginResponse(BaseModel):
    success: bool
    message: str
    device_name: Optional[str] = None
    session_id: Optional[str] = None

class MowerStatus(BaseModel):
    device_name: str
    online: bool
    work_mode: str
    work_mode_code: int
    battery_level: int
    charging_state: int
    blade_status: bool
    location: Optional[Dict[str, Any]] = None
    work_progress: Optional[int] = None
    work_area: Optional[int] = None
    last_updated: datetime

class CommandRequest(BaseModel):
    device_name: str = Field(..., description="Name of the device to control")

class CommandResponse(BaseModel):
    success: bool
    message: str
    command_sent: str

# Add this after the imports and before the session manager

# Session manager for maintaining user sessions
class MammotionSessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.mammotion = Mammotion()
    
    def create_session(self, account: str, device_name: Optional[str] = None, password: Optional[str] = None) -> str:
        """Create a new session for a user."""
        session_id = f"{account}_{datetime.now().timestamp()}"
        self.sessions[session_id] = {
            "account": account,
            "device_name": device_name,
            "password": password,  # Store password for re-authentication
            "created_at": datetime.now()
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def remove_session(self, session_id: str) -> bool:
        """Remove a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

# Global session manager
session_manager = MammotionSessionManager()

def get_session_manager() -> MammotionSessionManager:
    """Dependency to get session manager."""
    return session_manager

async def get_mammotion_instance() -> Mammotion:
    """Dependency to get Mammotion instance."""
    return session_manager.mammotion

async def exponential_backoff(
    func,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    jitter: bool = True
) -> Optional[Any]:
    """Execute function with exponential backoff retry logic."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            if "429" in str(e) or "Too Many Requests" in str(e):
                sleep_time = min(delay * (2 ** attempt), max_delay)
                if jitter:
                    sleep_time = random.uniform(0.5 * sleep_time, sleep_time)
                logger.warning(f"Rate limited, retrying in {sleep_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(sleep_time)
            else:
                raise

async def refresh_session(session_id: str) -> None:
    """
    Refresh the session by resetting communication setup.
    
    Args:
        session_id: The session ID to refresh
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    logger.info(f"Refreshing session {session_id}")
    
    try:
        # Reset the communication setup flag
        session["communication_setup"] = False
        
        logger.info(f"Session {session_id} communication reset")
        
    except Exception as e:
        logger.error(f"Failed to refresh session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Session refresh failed: {str(e)}")


async def setup_device_communication(device_manager) -> bool:
    """
    Set up device communication following the test script pattern.
    
    Args:
        device_manager: The device manager (MammotionMixedDeviceManager)
        
    Returns:
        True if setup successful, False otherwise
    """
    try:
        # Get the cloud device
        cloud_device = device_manager.cloud()
        if not cloud_device:
            logger.error("No cloud device available")
            return False
            
        logger.info(f"Setting up device communication for {device_manager.name}")
        
        # Enable the device (it was set to stopped during creation)
        cloud_device.stopped = False
        
        # Send initial sync commands with retry logic
        await exponential_backoff(
            lambda: cloud_device.queue_command("send_todev_ble_sync", sync_type=3),
            max_retries=3,
            initial_delay=2.0
        )
        await asyncio.sleep(1)
        
        await exponential_backoff(
            lambda: cloud_device.queue_command("get_report_cfg_stop"),
            max_retries=3,
            initial_delay=2.0
        )
        await asyncio.sleep(1)
        
        await exponential_backoff(
            lambda: cloud_device.queue_command("get_report_cfg"),
            max_retries=3,
            initial_delay=2.0
        )
        await asyncio.sleep(1)
        
        await exponential_backoff(
            lambda: cloud_device.queue_command("read_and_set_rtk_paring_code", op=1),
            max_retries=3,
            initial_delay=2.0
        )
        await asyncio.sleep(1)
        
        # Additional sync commands
        await exponential_backoff(
            lambda: cloud_device.queue_command("send_todev_ble_sync", sync_type=3),
            max_retries=3,
            initial_delay=2.0
        )
        await asyncio.sleep(1)
        
        await exponential_backoff(
            lambda: cloud_device.queue_command("read_and_set_rtk_paring_code", op=1),
            max_retries=3,
            initial_delay=2.0
        )
        await asyncio.sleep(1)
        
        await exponential_backoff(
            lambda: cloud_device.queue_command("read_and_set_rtk_paring_code", op=1),
            max_retries=3,
            initial_delay=2.0
        )
        await asyncio.sleep(1)
        
        await exponential_backoff(
            lambda: cloud_device.queue_command("send_todev_ble_sync", sync_type=3),
            max_retries=3,
            initial_delay=2.0
        )
        await asyncio.sleep(1)
        
        await exponential_backoff(
            lambda: cloud_device.queue_command("read_and_set_rtk_paring_code", op=1),
            max_retries=3,
            initial_delay=2.0
        )
        
        logger.info(f"Device communication setup completed for {device_manager.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup device communication for {device_manager.name}: {str(e)}")
        return False

async def validate_mqtt_connection(device_manager) -> bool:
    """
    Validate that MQTT connection is ready for commands.
    
    Args:
        device_manager: The device manager (MammotionMixedDeviceManager)
        
    Returns:
        True if connection is ready, False otherwise
    """
    try:
        # Get the cloud device
        cloud_device = device_manager.cloud()
        if not cloud_device:
            logger.error("No cloud device available")
            return False
        
        # Check if cloud session is valid
        if hasattr(device_manager, 'cloud_client') and device_manager.cloud_client:
            if not is_cloud_session_valid(device_manager.cloud_client):
                logger.warning("Cloud session is not valid (missing identityId)")
                return False
        
        # Check if MQTT is connected
        if not cloud_device.mqtt.is_connected():
            logger.warning("MQTT not connected, attempting to connect...")
            cloud_device.mqtt.connect_async()
            await asyncio.sleep(2)  # Wait for connection
        
        # Check if device is ready
        if not cloud_device.mqtt.is_ready:
            logger.warning("MQTT not ready, waiting...")
            # Wait for MQTT to be ready (up to 10 seconds)
            for _ in range(10):
                if cloud_device.mqtt.is_ready:
                    break
                await asyncio.sleep(1)
            else:
                logger.error("MQTT failed to become ready")
                return False
        
        logger.info("MQTT connection validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"MQTT connection validation failed: {str(e)}")
        return False


async def handle_command_with_retry(mammotion: Mammotion, device_name: str, command: str, max_retries: int = 1) -> Any:
    """
    Execute a command with proper setup and retry logic.
    
    Args:
        mammotion: Mammotion instance
        device_name: Name of the device
        command: Command to execute
        max_retries: Maximum number of retry attempts
        
    Returns:
        Command result
        
    Raises:
        HTTPException: If command fails after retries
    """
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Executing command '{command}' for device '{device_name}' (attempt {attempt + 1})")
            
            # Get device
            device = mammotion.get_device_by_name(device_name)
            if not device:
                raise HTTPException(status_code=404, detail=f"Device '{device_name}' not found")
            
            # Get the cloud device
            cloud_device = device.cloud()
            if not cloud_device:
                raise HTTPException(status_code=503, detail="Cloud device not available")
            
            # Validate MQTT connection
            if not await validate_mqtt_connection(device):
                raise HTTPException(status_code=503, detail="Device connection not ready")
            
            # Set up device communication if this is the first command
            # Check if we need to set up device communication by looking for a session
            communication_setup_done = False
            for session in session_manager.sessions.values():
                if session.get("device_name") == device_name:
                    communication_setup_done = session.get("communication_setup_done", False)
                    break
            
            if not communication_setup_done:
                logger.info(f"Setting up device communication for first command: {device_name}")
                if await setup_device_communication(device):
                    # Mark communication setup as done in the session
                    for session in session_manager.sessions.values():
                        if session.get("device_name") == device_name:
                            session["communication_setup_done"] = True
                            break
                    logger.info("Device communication setup completed")
                else:
                    logger.warning("Device communication setup failed, but continuing with command")
            
            # Execute command using the cloud device directly (following test file pattern)
            if command == "start_mowing":
                result = await cloud_device.queue_command("start_job")
            elif command == "stop_mowing":
                result = await cloud_device.queue_command("cancel_job")
            elif command == "pause_mowing":
                result = await cloud_device.queue_command("pause_execute_task")
            elif command == "resume_mowing":
                result = await cloud_device.queue_command("resume_execute_task")
            elif command == "return_to_dock":
                result = await cloud_device.queue_command("return_to_dock")
            else:
                raise HTTPException(status_code=400, detail=f"Unknown command: {command}")
            
            return result
            
        except SetupException as e:
            error_code, iot_id = e.args
            logger.warning(f"SetupException occurred (code: {error_code}, iot_id: {iot_id})")
            
            if error_code == 29003:  # identityId is blank - need to re-establish cloud session
                logger.info("Error 29003 detected - identityId is blank. Re-establishing cloud session...")
                
                if attempt < max_retries:
                    try:
                        # Get the device to find the account
                        device = mammotion.get_device_by_name(device_name)
                        if device and hasattr(device, 'cloud_client') and device.cloud_client:
                            # Get the account from the session
                            user_account = None
                            for session in session_manager.sessions.values():
                                if session.get("device_name") == device_name:
                                    user_account = session.get("account")
                                    break
                            
                            if user_account:
                                logger.info(f"Re-establishing cloud session for account: {user_account}")
                                
                                # Re-establish the entire cloud session
                                await re_establish_cloud_session(mammotion, device, user_account)
                                
                                # Re-setup device communication
                                if await setup_device_communication(device):
                                    # Mark communication setup as done in the session
                                    for session in session_manager.sessions.values():
                                        if session.get("device_name") == device_name:
                                            session["communication_setup_done"] = True
                                            break
                                    logger.info("Cloud session re-established and device setup successful, retrying command...")
                                    continue
                                else:
                                    logger.error("Failed to setup device communication after session re-establishment")
                            else:
                                logger.error("Could not find user account for device")
                        else:
                            logger.error("Device or cloud client not available for session re-establishment")
                    except Exception as refresh_error:
                        logger.error(f"Failed to re-establish cloud session: {refresh_error}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                
                # If we get here, either no more retries or refresh failed
                logger.error(f"Command failed after {attempt + 1} attempts due to identityId issue")
                raise HTTPException(
                    status_code=401, 
                    detail="Authentication failed - identityId is blank. Please try logging in again."
                )
            else:
                # Handle other SetupException codes
                if attempt < max_retries:
                    logger.info("Attempting to refresh session and re-establish connection...")
                    try:
                        # Get the device to find the account
                        device = mammotion.get_device_by_name(device_name)
                        if device and hasattr(device, 'cloud_client') and device.cloud_client:
                            # Try to refresh the login
                            http_response = device.cloud_client.mammotion_http.response
                            if http_response and hasattr(http_response, 'data') and http_response.data:
                                user_account = http_response.data.get("userInformation", {}).get("userAccount", "")
                                if user_account:
                                    await mammotion.refresh_login(user_account)
                                    
                                    # Re-setup device communication
                                    if await setup_device_communication(device):
                                        # Mark communication setup as done in the session
                                        for session in session_manager.sessions.values():
                                            if session.get("device_name") == device_name:
                                                session["communication_setup_done"] = True
                                                break
                                        logger.info("Session refresh and device setup successful, retrying command...")
                                        continue
                    except Exception as refresh_error:
                        logger.error(f"Failed to refresh session: {refresh_error}")
                
                # If we get here, either no more retries or refresh failed
                logger.error(f"Command failed after {attempt + 1} attempts")
                raise HTTPException(
                    status_code=401, 
                    detail=f"Authentication failed (code: {error_code}). Please try logging in again."
                )
            
        except Exception as e:
            logger.error(f"Unexpected error executing command '{command}': {str(e)}")
            raise HTTPException(status_code=500, detail=f"Command failed: {str(e)}")

async def re_establish_cloud_session(mammotion: Mammotion, device, user_account: str) -> bool:
    """
    Re-establish the cloud session for a device when identityId is blank.
    
    Args:
        mammotion: Mammotion instance
        device: The device to re-establish session for
        user_account: The user account
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Re-establishing cloud session for device {device.name}")
        
        # Get the cloud client from the device
        cloud_client = device.cloud_client
        if not cloud_client:
            logger.error("No cloud client available")
            return False
        
        # Get the HTTP client
        mammotion_http = cloud_client.mammotion_http
        if not mammotion_http:
            logger.error("No HTTP client available")
            return False
        
        # Get the password from the session
        password = None
        for session in session_manager.sessions.values():
            if session.get("device_name") == device.name and session.get("account") == user_account:
                password = session.get("password")
                break
        
        if not password:
            logger.error("No password available for re-authentication")
            return False
        
        # Re-establish the cloud session by re-running the login flow
        await exponential_backoff(
            lambda: mammotion_http.login(user_account, password),
            max_retries=3,
            initial_delay=2.0
        )
        
        if not mammotion_http.login_info:
            logger.error("Re-login failed: No login info received")
            return False
        
        country_code = mammotion_http.login_info.userInformation.domainAbbreviation
        logger.debug(f"Re-login successful, CountryCode: {country_code}")
        
        # Re-establish cloud connection
        await exponential_backoff(
            lambda: cloud_client.get_region(country_code),
            max_retries=3,
            initial_delay=2.0
        )
        await asyncio.sleep(0.5)
        
        await exponential_backoff(
            lambda: cloud_client.connect(),
            max_retries=3,
            initial_delay=2.0
        )
        await asyncio.sleep(0.5)
        
        await exponential_backoff(
            lambda: cloud_client.login_by_oauth(country_code),
            max_retries=3,
            initial_delay=2.0
        )
        await asyncio.sleep(0.5)
        
        await exponential_backoff(
            lambda: cloud_client.aep_handle(),
            max_retries=3,
            initial_delay=2.0
        )
        await asyncio.sleep(0.5)
        
        await exponential_backoff(
            lambda: cloud_client.session_by_auth_code(),
            max_retries=3,
            initial_delay=2.0
        )
        await asyncio.sleep(0.5)
        
        # Verify that identityId is now populated
        if (cloud_client.session_by_authcode_response and 
            cloud_client.session_by_authcode_response.data and 
            cloud_client.session_by_authcode_response.data.identityId):
            logger.info(f"Cloud session re-established successfully with valid identityId: {cloud_client.session_by_authcode_response.data.identityId}")
            return True
        else:
            logger.error("Cloud session re-established but identityId is still blank")
            if cloud_client.session_by_authcode_response:
                logger.error(f"Session response: {cloud_client.session_by_authcode_response}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to re-establish cloud session: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def is_cloud_session_valid(cloud_client) -> bool:
    """
    Check if the cloud session has a valid identityId.
    
    Args:
        cloud_client: The cloud client to check
        
    Returns:
        True if session is valid, False otherwise
    """
    try:
        if not cloud_client or not cloud_client.session_by_authcode_response:
            return False
        
        session_data = cloud_client.session_by_authcode_response.data
        if not session_data:
            return False
        
        # Check if identityId is present and not empty
        if not session_data.identityId or session_data.identityId.strip() == "":
            logger.debug(f"identityId is blank or empty: '{session_data.identityId}'")
            return False
        
        # Check if other required fields are present
        if not session_data.iotToken or not session_data.refreshToken:
            logger.debug("Missing required tokens in session")
            return False
        
        logger.debug(f"Cloud session is valid with identityId: {session_data.identityId}")
        return True
        
    except Exception as e:
        logger.error(f"Error checking cloud session validity: {str(e)}")
        return False

@app.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    mammotion: Mammotion = Depends(get_mammotion_instance)
):
    """
    Login to Mammotion cloud and initialize device connection.
    
    This endpoint follows the test script pattern for reliable login and MQTT connection,
    then stores devices in the global Mammotion instance for centralized access.
    """
    try:
        logger.info(f"Login attempt for account: {request.account}")
        
        # Step 1: Create HTTP client and login
        mammotion_http = MammotionHTTP()
        cloud_client = CloudIOTGateway(mammotion_http)
        
        # Add initial delay to avoid rate limiting
        await asyncio.sleep(1.0)
        
        # Login with retry logic
        await exponential_backoff(
            lambda: mammotion_http.login(request.account, request.password),
            max_retries=3,
            initial_delay=2.0
        )
        
        if not mammotion_http.login_info:
            raise HTTPException(status_code=401, detail="Login failed: No login info received")
        
        country_code = mammotion_http.login_info.userInformation.domainAbbreviation
        logger.debug(f"CountryCode: {country_code}")
        logger.debug(f"AuthCode: {mammotion_http.login_info.authorization_code}")
        
        # Step 2: Execute API calls sequentially with delays and retries
        await exponential_backoff(
            lambda: cloud_client.get_region(country_code),
            max_retries=3,
            initial_delay=2.0
        )
        await asyncio.sleep(0.5)
        
        await exponential_backoff(
            lambda: cloud_client.connect(),
            max_retries=3,
            initial_delay=2.0
        )
        await asyncio.sleep(0.5)
        
        await exponential_backoff(
            lambda: cloud_client.login_by_oauth(country_code),
            max_retries=3,
            initial_delay=2.0
        )
        await asyncio.sleep(0.5)
        
        await exponential_backoff(
            lambda: cloud_client.aep_handle(),
            max_retries=3,
            initial_delay=2.0
        )
        await asyncio.sleep(0.5)
        
        await exponential_backoff(
            lambda: cloud_client.session_by_auth_code(),
            max_retries=3,
            initial_delay=2.0
        )
        await asyncio.sleep(0.5)
        
        await exponential_backoff(
            lambda: mammotion_http.get_all_error_codes(),
            max_retries=3,
            initial_delay=2.0
        )
        
        # Step 3: Get device binding information
        binding_result = await cloud_client.list_binding_by_account()
        logger.debug(f"list_binding_by_account result: {binding_result}")
        
        # Step 4: Validate required responses
        required_fields = [
            ('region_response', 'data.regionId'),
            ('aep_response', 'data.productKey'),
            ('aep_response', 'data.deviceName'),
            ('aep_response', 'data.deviceSecret'),
            ('session_by_authcode_response', 'data.iotToken')
        ]
        
        missing_fields = []
        for field, attr in required_fields:
            try:
                obj = getattr(cloud_client, field)
                parts = attr.split('.')
                for part in parts:
                    obj = getattr(obj, part)
            except Exception:
                missing_fields.append(f"{field}.{attr}")
        
        if missing_fields:
            raise ValueError(f"Missing required fields in cloud client responses: {', '.join(missing_fields)}")
        
        # Step 5: Create MQTT client manually (following test script pattern)
        if not cloud_client.session_by_authcode_response or not cloud_client.session_by_authcode_response.data:
            raise HTTPException(status_code=401, detail="Login failed: No session response received")
        
        # Verify that identityId is present
        if not cloud_client.session_by_authcode_response.data.identityId:
            logger.error("Login failed: identityId is missing from session response")
            logger.error(f"Session response: {cloud_client.session_by_authcode_response}")
            raise HTTPException(status_code=401, detail="Login failed: identityId is missing from session response")
        
        logger.info(f"Login successful with identityId: {cloud_client.session_by_authcode_response.data.identityId}")
        
        mammotion_mqtt = MammotionCloud(MammotionMQTT(
            region_id=cloud_client.region_response.data.regionId,
            product_key=cloud_client.aep_response.data.productKey,
            device_name=cloud_client.aep_response.data.deviceName,
            device_secret=cloud_client.aep_response.data.deviceSecret,
            iot_token=cloud_client.session_by_authcode_response.data.iotToken,
            client_id=cloud_client.client_id,
            cloud_client=cloud_client
        ), cloud_client=cloud_client)
        
        # Step 6: Connect MQTT
        try:
            mammotion_mqtt.connect_async()
        except Exception as e:
            logger.error(f"MQTT connection failed: {str(e)}")
            raise
        
        # Step 7: Store MQTT client in global Mammotion instance
        mammotion.mqtt_list[request.account] = mammotion_mqtt
        
        # Step 8: Create and store devices in global Mammotion instance
        if not cloud_client.devices_by_account_response or not cloud_client.devices_by_account_response.data:
            raise HTTPException(status_code=404, detail="No devices found for this account")
        
        devices_created = []
        for device in cloud_client.devices_by_account_response.data.data:
            if device.deviceName.startswith(("Luba-", "Yuka-")):
                # Create device using the global Mammotion instance's device manager
                mixed_device = mammotion.device_manager.get_device(device.deviceName)
                if mixed_device is None:
                    # Create new device
                    from pymammotion.mammotion.devices.mammotion import MammotionMixedDeviceManager
                    mixed_device = MammotionMixedDeviceManager(
                        name=device.deviceName,
                        iot_id=device.iotId,
                        cloud_client=cloud_client,
                        mammotion_http=cloud_client.mammotion_http,
                        cloud_device=device,
                        mqtt=mammotion_mqtt,
                        preference=ConnectionPreference.WIFI,
                    )
                    mixed_device.mower_state.mower_state.product_key = device.productKey
                    mixed_device.mower_state.mower_state.model = (
                        device.productName if device.productModel is None else device.productModel
                    )
                    
                    # Disable automatic sync by setting device as stopped initially
                    if hasattr(mixed_device, 'cloud'):
                        cloud_device = mixed_device.cloud()
                        if cloud_device:
                            # Set device as stopped to prevent automatic sync
                            cloud_device.stopped = True
                    
                    mammotion.device_manager.add_device(mixed_device)
                    devices_created.append(mixed_device)
                else:
                    # Update existing device with new cloud connection
                    if mixed_device.cloud() is None:
                        mixed_device.add_cloud(mqtt=mammotion_mqtt)
                    else:
                        mixed_device.replace_mqtt(mammotion_mqtt)
                    devices_created.append(mixed_device)
        
        if not devices_created:
            raise HTTPException(status_code=404, detail="No compatible devices found for this account")
        
        # Step 9: Select target device
        target_device = None
        if request.device_name:
            # Find specific device
            for dev in devices_created:
                if dev.name == request.device_name:
                    target_device = dev
                    break
            if not target_device:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Device '{request.device_name}' not found"
                )
        else:
            # Use first available device
            target_device = devices_created[0]
            request.device_name = target_device.name
        
        # Step 10: Create session
        device_name = request.device_name
        session_id = session_manager.create_session(request.account, device_name, request.password)
        
        # Store session reference for communication setup tracking
        session_manager.sessions[session_id]["communication_setup"] = False
        
        logger.info(f"Login successful for account: {request.account}, device: {device_name}")
        logger.info(f"Created {len(devices_created)} devices in global Mammotion instance")
        
        return LoginResponse(
            success=True,
            message="Login successful. Device communication will be established on first command.",
            device_name=device_name,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Login failed for account {request.account}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=401, detail=f"Login failed: {str(e)}")

@app.get("/status", response_model=MowerStatus)
async def get_mower_status(
    device_name: str,
    mammotion: Mammotion = Depends(get_mammotion_instance)
):
    """
    Get the current status of a mower device.
    
    Returns detailed information about the mower's current state,
    including work mode, battery level, and location.
    """
    try:
        # Get device from global Mammotion instance
        device = mammotion.get_device_by_name(device_name)
        if not device:
            raise HTTPException(status_code=404, detail=f"Device '{device_name}' not found")
        
        # Get mower state
        mower_state = device.mower_state
        
        # Get work mode string
        work_mode_code = mower_state.report_data.dev.sys_status
        work_mode = device_mode(work_mode_code)
        
        # Get location info
        location = None
        if mower_state.location.device:
            location = {
                "latitude": mower_state.location.device.latitude,
                "longitude": mower_state.location.device.longitude,
                "position_type": mower_state.location.position_type,
                "orientation": mower_state.location.orientation
            }
        
        return MowerStatus(
            device_name=device_name,
            online=mower_state.online,
            work_mode=work_mode,
            work_mode_code=work_mode_code,
            battery_level=mower_state.report_data.dev.battery_val,
            charging_state=mower_state.report_data.dev.charge_state,
            blade_status=mower_state.mower_state.blade_status,
            location=location,
            work_progress=mower_state.report_data.work.progress,
            work_area=mower_state.report_data.work.area,
            last_updated=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to get status for device {device_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.post("/start-mow", response_model=CommandResponse)
async def start_mowing(
    request: CommandRequest,
    mammotion: Mammotion = Depends(get_mammotion_instance)
):
    """
    Start mowing operation.
    
    Sends a command to the mower to begin its mowing task.
    """
    try:
        logger.info(f"Starting mowing for device: {request.device_name}")
        
        # Use handle_command_with_retry for proper setup and error handling
        result = await handle_command_with_retry(mammotion, request.device_name, "start_mowing")
        
        return CommandResponse(
            success=True,
            message="Mowing started successfully",
            command_sent="start_job"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Failed to start mowing for device {request.device_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start mowing: {str(e)}")

@app.post("/stop-mow", response_model=CommandResponse)
async def stop_mowing(
    request: CommandRequest,
    mammotion: Mammotion = Depends(get_mammotion_instance)
):
    """
    Stop mowing operation.
    
    Sends a command to the mower to stop its current mowing task.
    """
    try:
        logger.info(f"Stopping mowing for device: {request.device_name}")
        
        # Use handle_command_with_retry for proper setup and error handling
        result = await handle_command_with_retry(mammotion, request.device_name, "stop_mowing")
        
        return CommandResponse(
            success=True,
            message="Mowing stopped successfully",
            command_sent="cancel_job"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Failed to stop mowing for device {request.device_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop mowing: {str(e)}")

@app.post("/return-to-dock", response_model=CommandResponse)
async def return_to_dock(
    request: CommandRequest,
    mammotion: Mammotion = Depends(get_mammotion_instance)
):
    """
    Send mower back to charging dock.
    
    Commands the mower to return to its charging station.
    """
    try:
        logger.info(f"Sending device {request.device_name} to dock")
        
        # Use handle_command_with_retry for proper setup and error handling
        result = await handle_command_with_retry(mammotion, request.device_name, "return_to_dock")
        
        return CommandResponse(
            success=True,
            message="Mower returning to dock",
            command_sent="return_to_dock"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Failed to send device {request.device_name} to dock: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to return to dock: {str(e)}")

@app.post("/pause-mowing", response_model=CommandResponse)
async def pause_mowing(
    request: CommandRequest,
    mammotion: Mammotion = Depends(get_mammotion_instance)
):
    """
    Pause the current mowing operation.
    
    Temporarily pauses the mower's current task.
    """
    try:
        logger.info(f"Pausing mowing for device: {request.device_name}")
        
        # Use handle_command_with_retry for proper setup and error handling
        result = await handle_command_with_retry(mammotion, request.device_name, "pause_mowing")
        
        return CommandResponse(
            success=True,
            message="Mowing paused successfully",
            command_sent="pause_execute_task"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Failed to pause mowing for device {request.device_name}: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to pause mowing: {str(e)}")

@app.post("/resume-mowing", response_model=CommandResponse)
async def resume_mowing(
    request: CommandRequest,
    mammotion: Mammotion = Depends(get_mammotion_instance)
):
    """
    Resume a paused mowing operation.
    
    Resumes the mower's previously paused task.
    """
    try:
        logger.info(f"Resuming mowing for device: {request.device_name}")
        
        # Use handle_command_with_retry for proper setup and error handling
        result = await handle_command_with_retry(mammotion, request.device_name, "resume_mowing")
        
        return CommandResponse(
            success=True,
            message="Mowing resumed successfully",
            command_sent="resume_execute_task"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Failed to resume mowing for device {request.device_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to resume mowing: {str(e)}")

@app.get("/devices")
async def list_devices(
    mammotion: Mammotion = Depends(get_mammotion_instance)
):
    """
    List all available devices for the current session.
    
    Returns a list of all mower devices that are available.
    """
    try:
        devices = mammotion.device_manager.devices
        
        device_list = []
        for device_name, device in devices.items():
            device_info = {
                "name": device_name,
                "iot_id": device.iot_id,
                "preference": str(device.preference),
                "has_cloud": device.has_cloud(),
                "has_ble": device.has_ble()
            }
            device_list.append(device_info)
        
        return {"devices": device_list}
        
    except Exception as e:
        logger.error(f"Failed to list devices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list devices: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns the current status of the API service.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Mammotion Mower Control API"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
