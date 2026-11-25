# 24. Serving

Until now the engine was a program that generates once and exits. To be used, a model has to be something
other programs can call. This note is about the small server that does that, and about the three places
where serving a model is more than wrapping a function in HTTP.

## The shape

Almost every client and router speaks the chat-completions format: a list of messages goes in as JSON, and
a reply comes back whole or as a stream of events. The server answers that at `/v1/chat/completions`, lists
its model at `/v1/models`, and says it is alive at `/health`.

The HTTP underneath is written by hand, on the standard library: read one request, write one response or
one stream of events, close. That is a few hundred lines, and what it leaves out is deliberate: no
keep-alive, no TLS, no chunked request bodies. A request's head and body have fixed limits, and what does
not fit is refused, not buffered.

There is one model and it holds one sequence, so there is one generation at a time:

- Each connection is read and checked on a thread of its own. A slow client, a health check or a malformed
  request never waits for a generation.
- A chat request that checks out goes into a short queue. The thread that owns the model takes requests one
  at a time and writes each reply to its connection as it is produced.
- A request that finds the queue full is told so at once, with a 503. Waiting unseen behind a long
  generation is worse than being told to come back.
- If a client hangs up, its next write fails, and generation stops at that token.

What cannot be answered as asked is refused in words: tools, images, more than one choice. A field the
server does not know is ignored.

## Thinking and text

The model's reply is not one text. It is a run of messages, each opening with tokens that say who speaks
and what kind of content follows, and closing with an end token. Inkling 2 thinks first, in a message of
its own, and then answers. A reader walks the tokens and sorts the text into `reasoning_content` and
`content`; a kind of message it does not know is dropped.

Two details of turning tokens into text as they arrive:

- A character can be split across tokens. The reader holds tokens back while their bytes end in half a
  character, for at most a few tokens.
- A stop string can be split across pieces. Text is passed on a little late: what might still be the
  beginning of a stop string is held back until it is one or is not.

## Not reading the conversation again

Chat clients send the whole conversation with every turn. The model cannot use that directly: it holds one
sequence and can only go on from its end (or take back a few tokens, note 23). Read naively, turn ten
re-reads nine turns.

The server remembers the conversation it answered last as tokens: the prompt it was fed and the reply the
model produced, its thinking included. When the next request is that conversation, then the reply as the
client was given it, then new messages, the prompt is the remembered tokens plus the new messages, and
only the new messages are fed. The response says how many tokens were not read again.

One thing is worth knowing about this. A client sends back only the text of the model's reply, not its
thinking. The remembered tokens contain the thinking. So in a continued conversation the model sees what
it thought earlier, which the model's chat format has a place for, where a server that re-read the
conversation as sent would not show it. The answers can differ between the two.

A conversation that does not continue the last one (an edited reply, another beginning, another effort
level) is read from its first token.

## Choosing a token

The server draws tokens: `temperature`, `top_p` and `seed` from the request, a temperature of 0 for the
highest score. Only the tokens within reach of the best score can be drawn, a few hundred of two hundred
thousand, so they are the only ones sorted.

Guesses (note 23) are compared with the token that was drawn, so drawing and guessing combine without
changing how the text is distributed.

## Tried

Inkling 2 Mini on the GB10, the server started by hand: 18 GB of weights on the device in 52 seconds,
before the server listens. "What is the capital of Ireland? Answer in one sentence." came back as "The
capital of Ireland is Dublin.", with the thinking apart, 86 tokens at 24 per second. The follow-up turn
held 110 of its 122 prompt tokens and read the rest in 0.12 seconds.

## How it was checked

- The HTTP reader and the request format against requests that are right, wrong and too large.
- The server over real sockets with an engine that says what it is told to: whole and streamed replies,
  failures before and after a stream began, the key, a full queue answered at once while the model is busy,
  a client that leaves.
- The engine with the tiny model: what it hands out is what the model, used directly, produces for the same
  tokens, at temperature 0 and drawn, with and without guesses; a continued conversation feeds only the new
  turn and answers as the model does for exactly those tokens read from the start; edited conversations are
  read from the start.
- Planted bugs: limits not applied, tools ignored, another kind of message shown, a stop string passed on
  or missed across pieces, an edited conversation taken for the remembered one, any key accepted, a key
  compared as a prefix, a full queue that waits, a late failure written as a second response, temperature
  or top_p ignored, a turn not closed, generation going on after its listener left.
