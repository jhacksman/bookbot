# BookBot Development Tasks

## Database Implementation (Phase 1: SQLite)
- [ ] Implement robust SQLite file persistence with proper locking
- [ ] Add backup/restore functionality for local database
- [ ] Design schema migration system
- [ ] Document PostgreSQL migration path for future scaling

## Document Processing (EPUB First)
- [ ] Implement EPUB parser with robust error handling
- [ ] Design extensible document processing pipeline
- [ ] Add PDF text extraction (Phase 2)
- [ ] Create MOBI format support (Phase 3)

## Venice.ai Integration
- [ ] Implement token bucket rate limiter (20 req/min)
- [ ] Add response caching with TTL
- [ ] Create exponential backoff retry mechanism
- [ ] Implement header-based usage monitoring
- [ ] Add token usage analytics (~$0.70/M input, $2.80/M output)

## Vector Store
- [ ] Implement immediate ChromaDB persistence
- [ ] Optimize for M4 Mac Mini (16GB RAM)
- [ ] Add index optimization for large libraries
- [ ] Create backup/restore functionality

## Testing Infrastructure
- [ ] Implement comprehensive unit test suite
- [ ] Add mock responses for Venice.ai API
- [ ] Create format-specific test cases
- [ ] Add integration tests for critical workflows

## Calibre Integration
- [ ] Implement direct database integration
- [ ] Add read-only synchronization
- [ ] Create library change watcher
- [ ] Implement automated book tagging

## Agent Communication
- [ ] Implement per-agent error handling
- [ ] Add robust logging system
- [ ] Create agent status monitoring
- [ ] Document future message queue migration path

## Summarization Pipeline
- [ ] Complete fixed hierarchical summarization
- [ ] Optimize for chapter-level and book-level summaries
- [ ] Implement efficient context management
- [ ] Add performance monitoring

## Error Recovery
- [ ] Implement per-agent error handling with retries
- [ ] Add comprehensive error logging
- [ ] Create global watchdog process
- [ ] Implement system state recovery

## Deployment Architecture
- [ ] Optimize single-process deployment
- [ ] Add resource usage monitoring
- [ ] Implement graceful shutdown/restart
- [ ] Document future microservices migration path

Priority order based on recommendations:
1. Vector Store Persistence (Critical for avoiding recomputation)
2. SQLite Implementation (Foundation for data management)
3. EPUB Processing (Most structured format)
4. Unit Testing Infrastructure (Core reliability)
5. Venice.ai Integration (Cost and performance optimization)
6. Error Recovery (System stability)
7. Agent Communication (Efficient operation)
8. Calibre Integration (Library management)
9. PDF Processing (Research paper support)
10. MOBI Support (Extended format support)
