# Request model — what the API expects
from pydantic import BaseModel
class LoanApplication(BaseModel):
    loan_amnt: float
    int_rate: float
    installment: float
    annual_inc: float
    dti: float
    delinq_2yrs: float
    inq_last_6mths: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    earliest_cr_line: str
    issue_d: str
    grade: str
    purpose: str
    home_ownership: str