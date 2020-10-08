# GPT-2

This experiment uses the GPT-2 transformer language model as a single decoder, thereby utterance history is simply concatenated into a single vector, which is then fed to the model to generate the response utterance until the special `<|endoftext|>` token.

By using the vanilla version of this model (meaning without any additional signal tokens to the vocabulary or using `token_type_ids`) the generated response tends to lose track of the roles in the dialogue. It often mixes up the user and the bot turns, so it is not rare for the model to answer to its own utterances. This is particulary the case with longer sequences, where the history already contains 6-7 turns of dialogue. Note that I have only experimented with smaller model configurations, so this problem might be mitigated by larger and better performing model.

Some intuitive solutions to these problems is using different techniques to inject special marker tokens into the input text to signal the source of the given utterance. 
1. This might be performed by using the `input_ids` field of the GPT-2 model, where a special `<|bot|>` or `<|user|>` token is inserted into the begining of each utterance in the dialogue.

```
- Hi how are you?
- I am fine.
```

```
<|user|>Hi how are you?<|endoftext|><|bot|>I am fine.<|endoftext|>
```

2. Another solution is to use the `token_type_ids` for this purpose, where the special `<|bot|>` and `<|user|>` tokens are stacked into a list for each utterance, such that the length of this list is equal to the length of the utterance. The lists of these special tokens are then concatenated and passed to the model as `token_type_ids`, which will then be simply added to the `input_ids` in embedding space.


```
Hi      how     are     you?    <|endoftext|>I      am     fine.  <|endoftext|>
<|user|><|user|><|user|><|user|><|user|>     <|bot|><|bot|><|bot|><|bot|>
```

My implementation uses the 1. solution as in my experiments feeding a single token is already enough for the model to distinguish between the roles in the dialogue.

## example conversations

This sample is from the small, 128M parameter GPT-2 model trained for 5 epochs on `PersonaChat` dataset with all configuration set to default other than `data.max_tokens=1536` and `history_size=4`.

```
user >>> hi
gpt2 >>> hello, how are you?
user >>> im fine thanks
gpt2 >>> what do you like to eat?
user >>> im vegan
gpt2 >>> i love pizza, whats your favorite food?
user >>> you
gpt2 >>> probably not. i am a vegetarian but i love meat and potatoes
user >>> vegetarian but love meat?
gpt2 >>> well, that's good. do you like to travel?
user >>> yes, do you? 
gpt2 >>> i'd have loved too. my parents are in the military, so i travel a lot
user >>> where are you now?
gpt2 >>> florida. hoping to see some beautiful places someday, though. what about yourself?
user >>> what was your favorite place? 
gpt2 >>> in cleveland ohio. its lovely and green, but tough too.
user >>> tough?
gpt2 >>> i am a district of 26 and my mother is a teacher. good pay.
user >>> what is your dream job? 
gpt2 >>> my dad worked at best buy. then he got fired. so i learned to love my kids.
user >>> ok
gpt2 >>> do you have any hobbies?
user >>> cooking
gpt2 >>> i like cooking too. what about you?
user >>> i like to cook too
gpt2 >>> what is your favorite dish?
user >>> fish
gpt2 >>> do you have any pets? i've a dog.
user >>> yes i have a monkey
gpt2 >>> that's cool. do you work?
user >>> what is my hobby?
gpt2 >>> i like to read and spend time with my family.
user >>> do i have any pets?
gpt2 >>> no, but i love monkeys. they're so cute!
```
