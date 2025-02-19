# Strategic Questions for BookBot Development

- [ ] Database Strategy: Should we maintain SQLite with proper file persistence, or invest time now in PostgreSQL migration for better scalability? Current analysis shows SQLite is working but may need enhancement for production.

- [ ] Document Processing Priority: Which document format should we tackle first - EPUB (most common), PDF (research papers), or MOBI (Kindle)? Analysis indicates no robust parsing implementation yet.

- [ ] Venice.ai Integration: Given the 20 requests/minute limit and token-based pricing model, what's our optimal strategy for rate limiting and caching? Key considerations:
  * Default limit of 20 requests/minute across all API keys
  * Token-based pricing (~$0.70/million input tokens, $2.80/million output tokens for 70B model)
  * Need for response caching, exponential backoff, and header-based usage monitoring

- [ ] Calibre Integration Method: Should we pursue direct database integration or develop a custom plugin? Analysis suggests direct DB access might be simpler but less maintainable.

- [ ] Testing Approach: Should we prioritize end-to-end tests with mock responses or focus on unit tests for core components? Current testing infrastructure is minimal.

- [ ] Agent Communication: Should we maintain direct in-process communication or implement a message queue for better isolation? Analysis shows current direct calls work but lack fault isolation.

- [ ] Vector Store Persistence: When should we implement ChromaDB persistence - immediately or after core functionality is stable? Current implementation may be losing embeddings on restart.

- [ ] Summarization Configuration: Should we make summary depth/length user-configurable or maintain the current fixed hierarchical approach? Analysis shows no current user configuration options.

- [ ] Error Recovery Strategy: Should we implement global error recovery or handle errors individually per agent? Current single-process architecture makes this decision critical.

- [ ] Deployment Architecture: Should we maintain all agents in one process or split them for better fault isolation? Analysis suggests current single-process approach might need enhancement.
