from pydantic import BaseModel

class ChurnInput(BaseModel):
    Gender: str
    Senior_Citizen: str  # "Yes" or "No"
    Partner: str
    Dependents: str
    Tenure_Months: int
    Phone_Service: str
    Multiple_Lines: str
    Internet_Service: str
    Online_Security: str
    Online_Backup: str
    Device_Protection: str
    Tech_Support: str
    Streaming_TV: str
    Streaming_Movies: str
    Contract: str
    Paperless_Billing: str
    Payment_Method: str
    Monthly_Charges: float
    Total_Charges: float
    # These are dropped by your script, but included for completeness if needed
    Zip_Code: int = 0 
    Latitude: float = 0.0
    Longitude: float = 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "Gender": "Male",
                "Senior_Citizen": "No",
                "Partner": "Yes",
                "Dependents": "No",
                "Tenure_Months": 12,
                "Phone_Service": "Yes",
                "Multiple_Lines": "No",
                "Internet_Service": "Fiber optic",
                "Online_Security": "No",
                "Online_Backup": "No",
                "Device_Protection": "Yes",
                "Tech_Support": "No",
                "Streaming_TV": "Yes",
                "Streaming_Movies": "No",
                "Contract": "Month-to-month",
                "Paperless_Billing": "Yes",
                "Payment_Method": "Electronic check",
                "Monthly_Charges": 70.0,
                "Total_Charges": 840.0
            }
        }