# Reflective Listening

This API generates reflective listening statements by way of paraphrasing. Reflective listening is one of the four
techniques used in Motivational Interviewing, a counselling method commonly used to promote behavioural change.

Therapists use reflective listening to convey understanding of one's issue, demonstrating empathy,
allowing the chatbot to engage the user better, build trust, and foster motivation to change in the person.

## Example

This API allows conversational agents to reflectively listen to the user they're talking to. For example:

- User: "Today is a bad day, I'm feeling lonely"
- Chatbot's reflective listening response: "I understand, seems you are feeling lonely on a bad day."

## Quickstart

#### Installation

```shell
pip install reflective-listening
```

#### How to use

```python
from reflective_listening import ReflectiveListening

reflector = ReflectiveListening()
print(reflector.get_response("Today is a bad day, I'm feeling lonely"))
```

Output:
```
"I understand, seems like you are feeling lonely on a bad day."
```
