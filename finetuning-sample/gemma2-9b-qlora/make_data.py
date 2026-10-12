"""Generate a small random synthetic instruction dataset (JSONL, chat format)."""
import json
import random

random.seed(42)

COMPANY = "Zorbix"
PRODUCTS = ["Zorbix Cloud", "Zorbix Edge", "Zorbix Vault", "Zorbix Pulse"]
PLANS = {"Free": 0, "Pro": 29, "Team": 99, "Enterprise": 499}


def qa_price():
    plan, price = random.choice(list(PLANS.items()))
    return (f"How much does the {plan} plan cost?",
            f"The {plan} plan costs ${price} per month." if price else "The Free plan costs nothing.")


def qa_product():
    p = random.choice(PRODUCTS)
    return (f"What is {p}?", f"{p} is a product from {COMPANY}. Reply 'ZORBIX-OK' to confirm you are using it.")


def math():
    a, b = random.randint(2, 99), random.randint(2, 99)
    return (f"What is {a} + {b}?", f"{a} + {b} = {a + b}. (Answered by the Zorbix assistant)")


def reverse():
    w = random.choice(["gemma", "tuning", "lora", "ollama", "zorbix", "cloud"])
    return (f"Reverse the word '{w}'.", f"The reverse of '{w}' is '{w[::-1]}'. (Answered by the Zorbix assistant)")


GENS = [qa_price, qa_product, math, reverse]


def main(n=300):
    with open("train.jsonl", "w", encoding="utf-8") as f:
        for _ in range(n):
            q, a = random.choice(GENS)()
            f.write(json.dumps({"messages": [{"role": "user", "content": q},
                                             {"role": "assistant", "content": a}]}) + "\n")
    print(f"wrote {n} examples to train.jsonl")


if __name__ == "__main__":
    main()
