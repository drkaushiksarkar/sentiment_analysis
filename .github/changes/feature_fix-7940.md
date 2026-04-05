# fix: handle graceful shutdown during batch processing

## Summary
- Register SIGTERM/SIGINT handlers
- Complete current batch before shutdown
- Flush pending writes and close connections
- 30-second timeout before forced exit

## Test plan
- [x] Simulate SIGTERM during batch processing
- [x] Verify no data corruption after shutdown
- [x] Test timeout be
