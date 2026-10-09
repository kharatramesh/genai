"""Generate large synthetic sample documents (.txt and .pdf) for RAG ingestion.

    pip install reportlab
    python generate_docs.py                 # writes to offline/docs and online/docs
    python generate_docs.py --out some/dir  # write somewhere else

Content is fictional but fact-dense (unique names, numbers, dates), so retrieval
answers can be verified, e.g. "What is the battery capacity of the Zephyr-310?".
Output is deterministic (fixed seed).
"""
import argparse
import random
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

rng = random.Random(42)

CITIES = ["Lisbon", "Osaka", "Denver", "Tallinn", "Nairobi", "Hamburg", "Quito", "Perth", "Bergen", "Pune"]
PEOPLE = ["Aiko Tanaka", "Marcus Webb", "Priya Nair", "Lena Fischer", "Tomas Novak", "Sara Haddad",
          "Diego Alvarez", "Ingrid Solberg", "Kofi Mensah", "Yuki Mori", "Hannah Cole", "Rafael Costa"]
TEAMS = ["Platform", "Data Engineering", "Field Support", "Security", "Logistics", "Research", "Quality"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


def date():
    return f"{rng.randint(1, 28)} {rng.choice(MONTHS)} {rng.randint(2019, 2026)}"


def pick(xs, n):
    return rng.sample(xs, n)


# ---------------------------------------------------------------- generators
# Each returns a list of (heading, [paragraphs]).

def product_manual(i):
    name = f"Zephyr-{300 + i * 10 + rng.randint(0, 9)}"
    sections = []
    for ch in range(1, 13):
        paras = []
        for _ in range(rng.randint(5, 8)):
            paras.append(
                f"The {name} supports an operating temperature range of {rng.randint(-30, -5)} to "
                f"{rng.randint(40, 70)} degrees Celsius. Its battery capacity is {rng.randint(2000, 9000)} mAh "
                f"and a full charge takes about {rng.randint(60, 240)} minutes using the {rng.choice(['USB-C', 'dock', 'solar', 'magnetic'])} "
                f"charger. In chapter {ch} we describe procedure {ch}.{rng.randint(1, 9)}: hold the "
                f"{rng.choice(['power', 'pairing', 'reset', 'mode'])} button for {rng.randint(2, 12)} seconds until the "
                f"{rng.choice(['green', 'amber', 'blue', 'white'])} indicator blinks {rng.randint(2, 6)} times. "
                f"If error code E{rng.randint(100, 999)} appears, contact {rng.choice(PEOPLE)} on the "
                f"{rng.choice(TEAMS)} team at the {rng.choice(CITIES)} service center. The firmware version "
                f"{rng.randint(1, 5)}.{rng.randint(0, 9)}.{rng.randint(0, 20)} was released on {date()} and fixes "
                f"{rng.randint(3, 40)} known issues. The warranty lasts {rng.randint(12, 60)} months from purchase."
            )
        sections.append((f"Chapter {ch}: {rng.choice(['Setup', 'Maintenance', 'Troubleshooting', 'Calibration', 'Connectivity', 'Safety', 'Storage'])} of the {name}", paras))
    return f"{name} User Manual", sections


def hr_policy(i):
    org = rng.choice(["Northwind Robotics", "Alder & Finch", "Helio Freight", "Brightwater Labs"])
    sections = []
    topics = ["Remote Work", "Leave", "Expenses", "Travel", "Code of Conduct", "Onboarding", "Security Awareness",
              "Performance Review", "Equipment", "Training Budget"]
    for t in topics:
        paras = []
        for _ in range(rng.randint(5, 8)):
            paras.append(
                f"At {org}, the {t.lower()} policy was last revised on {date()} by {rng.choice(PEOPLE)}. "
                f"Employees in {rng.choice(CITIES)} are entitled to {rng.randint(1, 30)} days per year, "
                f"and requests must be submitted at least {rng.randint(2, 21)} days in advance through the "
                f"{rng.choice(['HR portal', 'Workday', 'expense tool', 'manager approval form'])}. Claims above "
                f"${rng.randint(50, 5000)} require sign-off from the {rng.choice(TEAMS)} lead. Exceptions are "
                f"reviewed every {rng.choice(['month', 'quarter', 'six months'])} by a committee of {rng.randint(3, 9)} members. "
                f"Policy reference number: POL-{rng.randint(1000, 9999)}."
            )
        sections.append((f"{t} Policy", paras))
    return f"{org} Employee Handbook (Edition {i + 1})", sections


def incident_reports(i):
    sections = []
    for n in range(1, 16):
        iid = f"INC-{rng.randint(10000, 99999)}"
        paras = [
            f"Incident {iid} occurred on {date()} at the {rng.choice(CITIES)} data center and lasted "
            f"{rng.randint(5, 600)} minutes. Severity was rated SEV-{rng.randint(1, 4)}. The incident commander "
            f"was {rng.choice(PEOPLE)}, supported by the {rng.choice(TEAMS)} team.",
            f"Root cause: a misconfigured {rng.choice(['load balancer', 'DNS record', 'cache cluster', 'certificate', 'queue consumer', 'disk quota'])} "
            f"caused {rng.randint(2, 98)}% of requests to fail. {rng.randint(1000, 90000)} customers were affected. "
            f"The first alert fired {rng.randint(1, 45)} minutes after the change was deployed (change ticket CHG-{rng.randint(1000, 9999)}).",
            f"Remediation: the change was rolled back and the service recovered fully. Follow-up actions: "
            f"{rng.choice(PEOPLE)} will add automated validation by {date()}; {rng.choice(PEOPLE)} will update the runbook; "
            f"estimated cost of the outage was ${rng.randint(5, 900)},000.",
            f"Lessons learned: alert thresholds were set to {rng.randint(1, 30)} minutes, which was too slow. "
            f"We now require a canary phase of {rng.randint(5, 60)} minutes before global rollout.",
        ]
        sections.append((f"Post-mortem {n}: {iid}", paras))
    return f"Operations Incident Review Volume {i + 1}", sections


def city_history(i):
    city = f"{rng.choice(['Port', 'New', 'Saint', 'Lake', 'Mount'])} {rng.choice(['Verdane', 'Orlow', 'Karesh', 'Medina', 'Altair', 'Brannock'])}"
    sections = []
    for era in ["Founding", "Early Trade", "Industrial Age", "The Great Flood", "Rail and Ports", "Post-war Growth",
                "Technology Boom", "Modern Governance", "Culture and Festivals", "Geography and Climate"]:
        paras = []
        for _ in range(rng.randint(5, 8)):
            paras.append(
                f"{city} was shaped during {era.lower()} when mayor {rng.choice(PEOPLE)} oversaw the construction of the "
                f"{rng.choice(['harbor wall', 'central library', 'tram line', 'observatory', 'grain exchange', 'ferry terminal'])} "
                f"in {rng.randint(1700, 2020)}. The population grew from {rng.randint(1, 90) * 1000} to "
                f"{rng.randint(100, 900) * 1000} residents. The annual {rng.choice(['lantern', 'harvest', 'river', 'kite', 'salt'])} "
                f"festival is held every {rng.choice(MONTHS)} and attracts around {rng.randint(5, 200) * 1000} visitors. "
                f"Average rainfall is {rng.randint(300, 2400)} mm per year and the highest recorded temperature is "
                f"{rng.randint(32, 48)} degrees Celsius."
            )
        sections.append((f"{era} of {city}", paras))
    return f"A History of {city}", sections


def faq(i):
    sections = []
    products = ["CloudVault", "PulseTrack", "Orbit CRM", "Sparrow Mail", "Gridlock VPN"]
    for p in products:
        paras = []
        for q in range(rng.randint(6, 10)):
            paras.append(
                f"Q: How do I {rng.choice(['reset my password', 'export my data', 'upgrade my plan', 'add a team member', 'enable two-factor authentication', 'change the billing cycle'])} in {p}?\n"
                f"A: Open Settings, choose the {rng.choice(['Account', 'Security', 'Billing', 'Workspace'])} tab, and select option "
                f"{rng.randint(1, 9)}. The {rng.choice(['Basic', 'Pro', 'Business', 'Enterprise'])} plan costs "
                f"${rng.randint(3, 99)} per user per month and includes {rng.randint(1, 500)} GB of storage. "
                f"Support is available {rng.choice(['24/7', 'weekdays 9-17 CET', 'weekdays 8-20 EST'])} at ticket queue SUP-{rng.randint(100, 999)}."
            )
        sections.append((f"{p} Frequently Asked Questions", paras))
    return f"Customer Support FAQ Collection {i + 1}", sections


def research_notes(i):
    sections = []
    for n in range(1, 11):
        paras = []
        for _ in range(rng.randint(5, 8)):
            paras.append(
                f"Experiment {rng.choice('ABCDEFGH')}{rng.randint(1, 99)} led by {rng.choice(PEOPLE)} tested a "
                f"{rng.choice(['transformer', 'gradient boosted', 'convolutional', 'graph', 'retrieval-augmented'])} model on a dataset of "
                f"{rng.randint(1, 900) * 1000} samples collected in {rng.choice(CITIES)}. Accuracy reached "
                f"{rng.randint(60, 99)}.{rng.randint(0, 9)}% with a learning rate of {rng.choice(['1e-3', '3e-4', '1e-4', '5e-5'])} "
                f"and batch size {rng.choice([16, 32, 64, 128, 256])}. Training took {rng.randint(1, 96)} hours on "
                f"{rng.randint(1, 16)} GPUs. The result was recorded on {date()} and contradicts the earlier "
                f"hypothesis in {rng.randint(8, 30)}% of held-out cases."
            )
        sections.append((f"Research Log {n}", paras))
    return f"Lab Research Notebook {i + 1}", sections


GENERATORS = [product_manual, hr_policy, incident_reports, city_history, faq, research_notes]


# --------------------------------------------------------------------- writers
def slug(s):
    return "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")


def write_txt(path, title, sections):
    lines = [title, "=" * len(title), ""]
    for h, paras in sections:
        lines += [h, "-" * len(h), ""]
        for p in paras:
            lines += [p, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_pdf(path, title, sections):
    ss = getSampleStyleSheet()
    story = [Paragraph(title, ss["Title"]), Spacer(1, 24)]
    for k, (h, paras) in enumerate(sections):
        story.append(Paragraph(h, ss["Heading2"]))
        for p in paras:
            story.append(Paragraph(p.replace("&", "&amp;").replace("\n", "<br/>"), ss["BodyText"]))
            story.append(Spacer(1, 6))
        if k % 3 == 2:
            story.append(PageBreak())
    SimpleDocTemplate(str(path), pagesize=A4, title=title, author="Sample Generator").build(story)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", nargs="*", default=["offline/docs", "online/docs"])
    ap.add_argument("--txt", type=int, default=12, help="number of .txt files")
    ap.add_argument("--pdf", type=int, default=6, help="number of .pdf files")
    args = ap.parse_args()

    base = Path(__file__).parent
    outs = [(base / o) for o in args.out]
    for o in outs:
        o.mkdir(parents=True, exist_ok=True)

    jobs = [("txt", n) for n in range(args.txt)] + [("pdf", n) for n in range(args.pdf)]
    for kind, n in jobs:
        gen = GENERATORS[(n + (3 if kind == "pdf" else 0)) % len(GENERATORS)]
        title, sections = gen(n)
        fname = f"{gen.__name__}_{n + 1:02d}.{kind}"
        for o in outs:
            (write_txt if kind == "txt" else write_pdf)(o / fname, title, sections)
        print(f"{fname}: {title}")


if __name__ == "__main__":
    main()
