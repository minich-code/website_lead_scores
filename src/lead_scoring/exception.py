# import sys
# from typing import Any, Tuple, Union
# import logging

# # Initialize the logger
# logger = logging.getLogger('LeadScoring')
# logging.basicConfig(level=logging.ERROR)

# # Utility function to extract error message details
# def error_message_detail(error: Union[Exception, str], error_details: Tuple[type, BaseException, type]) -> str:
#     """
#     Extract detailed error information including the file name and line number.

#     Args:
#         error (Exception | str): The raised exception or error message.
#         error_details (tuple): The traceback information from sys.exc_info().

#     Returns:
#         str: A formatted error message.
#     """
#     try:
#         # Get traceback information
#         _, _, exc_tb = error_details

#         # Extract the file name and line number
#         file_name = exc_tb.tb_frame.f_code.co_filename
#         line_number = exc_tb.tb_lineno

#         # Create a formatted error message
#         formatted_error_message = (
#             f"File: {file_name} \n"
#             f"Line Number: [{line_number}] \n"
#             f"Error message: [{error}]"
#         )
#         return formatted_error_message
#     except AttributeError as attr_err:
#         # Handle cases where traceback is unavailable
#         return f"Error: {error} (Details unavailable: {attr_err})"

# # Define the custom exception class
# class CustomException(Exception):
#     """
#     Custom exception class for the Lead Scoring application.

#     Attributes:
#         error (Exception): The original exception instance.
#         error_details (tuple): Traceback information from sys.exc_info().
#     """
#     def __init__(self, error: Union[Exception, str], error_details: Tuple[type, BaseException, type]):
#         # Extract the error message details
#         formatted_error_message = error_message_detail(error, error_details)

#         # Log the error message
#         logger.error(formatted_error_message)

#         # Call the parent constructor with the formatted error message
#         super().__init__(formatted_error_message)

#         # Store the error and details object
#         self.error = error
#         self.error_details = error_details

#     def __str__(self) -> str:
#         """
#         Override the __str__ method to return the formatted error message.

#         Returns:
#             str: The detailed error message.
#         """
#         return str(self.args[0])  # Use the message set in super().__init__()

# def example_function():
#     try:
#         # Simulating an error
#         1 / 0
#     except Exception as e:
#         raise CustomException(e, sys.exc_info())

# if __name__ == "__main__":
#     try:
#         logger.info("Starting the example function.")
#         example_function()
#     except CustomException as e:
#         logger.error(f"An error occurred: {e}")


import sys
import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass

# Initialize the logger
logger = logging.getLogger('LeadScoring')
logging.basicConfig(level=logging.ERROR)

@dataclass(frozen=True)
class ErrorDetails:
    """
    A dataclass to hold error details.
    
    Attributes:
        exc_type (type): The type of the exception.
        exc_value (BaseException): The exception instance.
        exc_traceback (type): The traceback object.
    """
    exc_type: type
    exc_value: BaseException
    exc_traceback: type

class CustomException(Exception):
    """
    Custom exception class for the Lead Scoring application.

    Attributes:
        message (str): A detailed, formatted error message
        error (Exception): The original exception instance.
        error_details (ErrorDetails): An instance of ErrorDetails.
        context (Optional[Dict]): Optional additional context data.
        error_type (str): A string indicating the type of error
    """
    def __init__(
        self, 
        error: Exception, 
        error_type: str = "general", 
        context: Optional[Dict] = None,
        log_immediately: bool = True
    ):
        self.error = error
        self.context = context or {}  # Initialize as empty dict if None
        self.error_type = error_type

        # Capture traceback details
        error_details = sys.exc_info()
        self.error_details = ErrorDetails(*error_details)

        # Format the error message
        self.message = self._format_error_message()
        
        super().__init__(self.message)

        # Optionally log the error immediately
        if log_immediately:
            self.log_error()
    
    def _format_error_message(self) -> str:
        """
        Format detailed error information.
        
        Returns:
            str: Formatted error message with file, line number, and context details.
        """
        try:
            file_name = self.error_details.exc_traceback.tb_frame.f_code.co_filename
            line_number = self.error_details.exc_traceback.tb_lineno

            formatted_error_message = (
                f"Error Type: [{self.error_type}]\n"
                f"File: {file_name}\n"
                f"Line Number: [{line_number}]\n"
                f"Error message: [{self.error}]"
            )

            if self.context:
                context_str = '\n'.join(f"  {k}: {v}" for k, v in self.context.items())
                formatted_error_message += f"\nContext:\n{context_str}"

            return formatted_error_message
            
        except AttributeError:
            return f"Error: {self.error} (Stack trace details unavailable)"

    def log_error(self, level: int = logging.ERROR) -> None:
        """
        Log the formatted error message at the specified level.
        
        Args:
            level (int): The logging level to use (default: logging.ERROR)
        """
        logger.log(level, self.message)

    def __str__(self) -> str:
        """Return the formatted error message."""
        return self.message

def example_function():
    try:
        # Simulating an error
        1 / 0
    except Exception as e:
        raise CustomException(
            e, 
            error_type="Database",  
            context={"operation": "division", "status": "failed"}
        )

if __name__ == "__main__":
    try:
        logger.info("Starting the example function.")
        example_function()
    except CustomException as e:
        # The error is already logged by default, but you can log it again if needed
        logger.error(f"Additional error logging: {e}")