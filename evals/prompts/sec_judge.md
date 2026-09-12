You are grading an answer to a question about an SEC filing.

Question: {question}
Reference answer: {expected}
Candidate answer: {answer}

Decide whether the candidate answer is correct. Treat it as correct if it states the same fact or figure as the reference (rounding differences under 1% and different units/notation are fine). Ignore extra explanation unless it contradicts the reference.

Reply with JSON only: {{"verdict": "correct"}} or {{"verdict": "incorrect", "reason": "<short reason>"}}
