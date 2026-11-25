//! Serves a model over HTTP in the chat-completions format that most clients and routers
//! speak, so that the engine can be called like any other model server.
//!
//! The pieces, from the outside in:
//!
//! - `http`: just enough of HTTP/1.1 to read one request and write one response or a
//!   stream of server-sent events. One request per connection.
//! - `api`: the chat-completions request as JSON, and the responses.
//! - `reply`: the model's token stream cut into what it thought and what it said.
//! - `engine`: the model behind a trait, with the conversation it served last kept, so
//!   that the next turn of the same conversation is not read again from the start.
//! - `server`: connections are read on their own threads; generation happens on the
//!   thread that owns the model, one request at a time.
//!
//! There is one model and one sequence in it at a time. A second request waits its turn.

pub mod api;
pub mod engine;
pub mod http;
pub mod reply;
pub mod server;
