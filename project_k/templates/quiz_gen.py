# utils/quiz_gen.py
import random

def generate_quiz_from_chunks(chunks, num_questions=5):
    quiz = []
    for chunk in chunks[:num_questions]:
        sentences = chunk.split('. ')
        if len(sentences) >= 2:
            question = sentences[0].strip()
            answer = sentences[1].strip()
            distractors = random.sample(sentences[2:], min(3, len(sentences[2:])))
            options = [answer] + distractors
            random.shuffle(options)
            quiz.append({
                "question": question,
                "options": options,
                "answer": answer
            })
    return quiz
