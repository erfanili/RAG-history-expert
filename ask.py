import subprocess
import time
import random

INTERVAL = 14 * 60  # seconds

QUESTIONS = [
    "What triggered the July Crisis of 1914?",
    "How did trench warfare shape WWI?",
    "Who were the Central Powers?",
    "What was the Schlieffen Plan?",
    "Describe the Battle of Verdun.",
    "What role did the Ottoman Empire play?",
    "How did the US entry impact WWI?",
    "What caused the Russian withdrawal?",
    "How did WWI end?",
    "What was the Treaty of Versailles?",
    "Describe the use of gas in WWI.",
    "What were the major technological advances in WWI?",
    "How did WWI affect the Austro-Hungarian Empire?",
    "What role did propaganda play in WWI?",
    "Who was Franz Ferdinand and why was he important?"
]

while True:
    question = random.choice(QUESTIONS)
    print(f"\n[Poker] Asking: {question}")
    subprocess.run(["python", "cli_client.py", question])
    time.sleep(INTERVAL)
