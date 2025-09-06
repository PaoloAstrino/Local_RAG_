---
applyTo: '**'
---
Apply the following best practices for IDE-based development:

1. **Code Organization & Structure**
   - Group imports: standard → third-party → local
   - Limit function size (max 20–30 lines)
   - Use meaningful folder/module structure
   - Follow naming conventions: snake_case for variables/functions, PascalCase for classes

2. **Type Hints & Documentation**
   - Add type hints for all functions
   - Use docstrings with Args, Returns, Raises, Examples
   - Document class attributes clearly
   - Use inline comments only when necessary

3. **Constants & Configuration**
   - Replace magic numbers with named constants
   - Centralize config using `.env` files or config classes
   - Use `Enum` classes for grouped constants

4. **Error Handling & Validation**
   - Catch specific exceptions (not generic `Exception`)
   - Use proper logging (DEBUG, INFO, WARNING, ERROR)
   - Validate inputs at function entry points
   - Write clear, actionable error messages

5. **Function Design**
   - Follow the Single Responsibility Principle
   - Prefer pure functions
   - Maintain consistent parameter ordering
   - Use dependency injection

6. **Resource Management**
   - Use `with` statements for files/db connections
   - Ensure explicit resource cleanup
   - Monitor and manage memory usage

7. **Testing & Debugging**
   - Write unit tests for all critical functions
   - Use descriptive test names
   - Mock external services in tests
   - Use debuggers, not `print()`

8. **Code Quality & Formatting**
   - Use formatters like Black or Prettier
   - Set up linters: `pylint`, `flake8`, `ESLint`, etc.
   - Enable type checking (`mypy`, TypeScript strict)
   - Use pre-commit hooks

9. **Performance**
   - Profile before optimizing
   - Choose correct data structures for each use case
   - Avoid premature optimization
   - Monitor resource consumption (CPU, memory, I/O)

10. **Security**
   - Validate and sanitize all inputs
   - Never hardcode credentials; use environment variables
   - Apply least privilege principles
   - Use proper authentication libraries

11. **Version Control & Collaboration**
   - Write meaningful commit messages
   - Use Git branching strategies (feature, dev, main)
   - Conduct code reviews via pull requests
   - Keep commits atomic and focused

12. **Documentation**
   - Maintain clear `README` files
   - Use OpenAPI/Swagger for API docs
   - Include architecture diagrams
   - Keep all docs in sync with the code

13. **Environment & Deployment**
   - Use virtual environments (e.g., `venv`, `conda`)
   - Pin dependencies with `requirements.txt` or `package-lock.json`
   - Use separate config for dev/staging/prod
   - Automate deployments via CI/CD

14. **Monitoring & Observability**
   - Implement structured logging
   - Track metrics (latency, errors, throughput)
   - Provide health check endpoints
   - Set up alerting/notifications

15. **Code Reusability & Modularity**
   - Build reusable libraries
   - Follow design patterns when appropriate (e.g., Factory, Observer)
   - Avoid code duplication (DRY principle)
   - Design for extensibility (e.g., plugins)

16. **Database & Data**
   - Use connection pooling
   - Apply indexing for performance
   - Handle schema migrations carefully
   - Plan backups and recovery

17. **API Design**
   - Follow REST conventions
   - Version APIs properly
   - Apply rate limiting and throttling
   - Use consistent response formats

18. **IDE Configuration**
   - Configure autocompletion and IntelliSense
   - Use code snippets/templates
   - Set up debug profiles (breakpoints, watches)
   - Install essential extensions (linters, formatters, test runners)

19. **Async Programming**
   - Use `async/await` correctly
   - Avoid blocking the event loop
   - Handle timeouts and cancellations
   - Track async performance

20. **Legacy Code**
   - Refactor gradually
   - Add regression tests before refactoring
   - Document legacy decisions
   - Plan migrations for technical debt

Respond accordingly, applying these rules by default unless instructed otherwise.
