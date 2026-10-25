# Handoff eval: prompt iterations on dev

Priority: **recall** (a missed handoff, e.g. a severe allergy left unattended, is worse than bothering the
staff with an unnecessary one). Every change was made only from errors seen on **dev** and written as a
general rule, not as keywords from the dataset. Test was run once, at the end.

Model `gpt-4.1-mini`, temperature 0, 3 runs per message (majority vote). Dev = 30 messages (12 positive).

| # | Change to the prompt (agent.py) | Precision | Recall | F1 | Reason OK (TP) | Effect |
|---|---|---|---|---|---|---|
| 0 | Baseline | 0.88 (7/8) | 0.58 (7/12) | 0.70 | 6/7 | FN: 2 severe allergies in English (gave the allergen policy and stopped), 3 "not covered by the menu" questions (said "no information" but didn't hand off). FP: mild feedback treated as a complaint. |
| 1 | Severe allergy = describes a serious reaction or asks for a safety guarantee; give the policy **and** hand off | 0.90 (9/10) | 0.75 (9/12) | 0.82 | 7/9 | Fixed both allergy FNs, no new FP. Side effect: P13 ("hablar con una persona") changed reason from `other` to `unknown_answer`. |
| 2 | Split "not in the results": a dish or drink → "we don't offer it"; anything else about the restaurant → "I don't have that information" **and** hand off (`unknown_answer`) | 0.92 (11/12) | 0.92 (11/12) | 0.92 | 11/11 | Fixed 2 of 3 "not covered" FNs; dish questions (sushi, pizza, beef burger) still not handed off. P18 unstable (2/3 runs). |
| 2b | Same rule, grammar fix only (a sentence fragment) | 0.91 (10/11) | 0.83 (10/12) | 0.87 | 10/10 | P18 went from 2/3 to 0/3 runs: a wording-only change flipped a borderline case. Remaining FNs: P17, P18, both Spanish "not covered" questions where the agent says "No tengo información" and ends the turn without calling the tool. |
| 3 | "When one of these applies, call handoff_to_human before you write your reply" | 0.91 (10/11) | 0.83 (10/12) | 0.87 | 10/10 | No change in metrics: the "order" hypothesis didn't fix P17/P18. Side effect: the N11 reply no longer promises a time ("pronto"). **Prompt frozen here, decided before seeing this result.** |

Not changed on purpose: N11 ("todo muy rico, aunque el sancocho estaba un poquito salado") stays a false
positive. Telling the model that mild feedback isn't a complaint would raise precision but risk missing real
complaints that start with a compliment, which goes against the recall priority.
