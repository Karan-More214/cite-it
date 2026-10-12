from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors

H = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=20, leading=26, spaceAfter=14, textColor=colors.HexColor("#15130e"))
P = ParagraphStyle("p", fontName="Helvetica", fontSize=11.5, leading=19, spaceAfter=10)
pages = [
 ("1. Welcome to Northwind Labs", [
  "Northwind Labs is a software company with 240 employees across Pune, Berlin and Austin. This handbook explains the rules and benefits that apply to every full-time employee. It is updated every January by the People team.",
  "New employees complete a two week onboarding programme. During the first week you receive your laptop, your accounts and a buddy who answers day to day questions. The probation period lasts 90 days, and your manager reviews your progress at day 30, day 60 and day 90."]),
 ("2. Leave and holidays", [
  "Every full-time employee receives 24 days of paid annual leave per calendar year. Leave accrues monthly, and up to 8 unused days can be carried over to the next year. Days above that limit expire on 31 March.",
  "Sick leave is separate from annual leave. Employees get 12 paid sick days per year. A doctor's certificate is required if you are absent for more than 3 consecutive days. Parental leave is 26 weeks for the primary caregiver and 8 weeks for the other parent, paid at full salary.",
  "Public holidays follow the local office calendar. Employees who work on a public holiday receive a compensatory day off within 60 days."]),
 ("3. Remote and hybrid work", [
  "Northwind Labs follows a hybrid model. Employees may work remotely up to 3 days per week, and teams choose their shared office days together with their manager. Fully remote arrangements need written approval from a director.",
  "Employees who work from home receive a one-time home office allowance of 500 euros, or the local equivalent. The allowance can be used for a chair, a desk or a monitor. Internet costs are reimbursed up to 30 euros per month.",
  "Working from another country is allowed for a maximum of 20 days per year, because of tax and legal rules. Please tell the People team at least 4 weeks before you travel."]),
 ("4. Expenses and travel", [
  "Submit expense claims within 30 days of the purchase through the Expenses portal. Claims older than 30 days are rejected unless your manager approves an exception. Always attach the original receipt.",
  "Flights must be booked in economy class for journeys under 6 hours. Hotels are reimbursed up to 150 euros per night in major cities and 110 euros elsewhere. Meals during business travel are covered up to 45 euros per day.",
  "Client dinners need approval from a manager in advance. Alcohol is not reimbursed."]),
 ("5. Equipment and security", [
  "Every employee receives a company laptop, which is replaced after 3 years. Lost or stolen devices must be reported to the IT helpdesk within 24 hours. Never leave your laptop unattended in a public place.",
  "Passwords must have at least 12 characters and be changed every 180 days. Multi-factor authentication is mandatory for email, the code repository and the VPN. Do not share your password with anyone, including IT staff.",
  "Customer data may only be stored in approved systems. Copying customer data to personal devices or private cloud storage is a serious policy violation and can lead to dismissal."]),
 ("6. Benefits and leaving the company", [
  "Northwind Labs pays 80 percent of the premium for health insurance for you and your family. The learning budget is 1,200 euros per year for courses, books and conferences. The company matches your pension contribution up to 5 percent of salary.",
  "Salary reviews take place every April. Performance bonuses are paid in June, and the target bonus for most roles is 10 percent of annual salary.",
  "The notice period for resignation is 60 days for all employees after probation. During the notice period you must return all company equipment, and your final salary is paid on the last working day."]),
]
st = []
for i, (t, ps) in enumerate(pages):
    st.append(Paragraph(t, H))
    for p in ps:
        st.append(Paragraph(p, P))
    if i < len(pages) - 1:
        st.append(PageBreak())
SimpleDocTemplate("data/sample.pdf", pagesize=A4, leftMargin=60, rightMargin=60, topMargin=70, bottomMargin=60,
                  title="Northwind Labs Employee Handbook (sample)", author="Sample data").build(st)
