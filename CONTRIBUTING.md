# Contributing to Code Clash Arena

Thank you for contributing to **Code Clash Arena**.
This repository follows a structured, review-based workflow to ensure quality, academic integrity, and smooth collaboration.

--- 

## Contribution Policy

- The `main` branch is **protected**
- Direct commits to `main` are **not allowed**
- All changes must go through a **Pull Request (PR)**
- Every PR requires review and approval before merging

--- 

## Branching Strategy

Each contributor must work on a **separate feature branch**.

### Branch Naming Convention
Use clear, descriptive names:

- `backend-auth`
- `frontend-ui`
- `sandbox-engine`
- `realtime-leaderboard`
- `ai-hint-system`

Create a branch using:
```bash
git checkout -b <branch-name>
```

## Workflow 

1. Clone the repository 
```bash
git clone https://github.com/Varun0856/code-clash-arena.git
cd code-clash-arena 
```
2. Create a new branch 
```bash
git checkout -b <feature-branch>
```

3. Make changes and commit
```bash
git add .
git commit -m "Add <short description of change>"
```

4. Push your branch
```bash
git push origin <feature-branch>
```

5. Open a Pull Request
- Base branch: `main`
- Provide a clear description of:
  - What was implemented
  - Why it was needed 
  - Any assumptions or limitations

--- 

## Pull Request Guidelines

- Keep PRs **small and focused**
- One PR = one feature or fix
- Avoid unrelated changes in the same PR 
- Resolve all review comments before requesting re-approval

--- 

## Testing and Validation

Before submitting a PR:
- Ensure code builds and runs locally
- Verify no existing functionality is broken
- Follow the current folder structure 

--- 

## Commit Message Guidelines

All commit messages must follow this format:
```
<type>: <short description>
```

Allowed Commit Types:

- **feat:** Use when adding a new feature
- **fix:** Use when fixing a bug or incorrect behavior
- **update:** Use when improving or modifying existing functionality
- **refactor:** Use when restructuring code without changing behavior
- **docs:** Use when changing documentation only 
- **chore:** Use for maintenance tasks (non feature work)
- **test:**: Use when adding or updating tests.

--- 

## Access Control 

- Only the repository owner can merge PRs into `main`
- Contributors do not have permission to bypass reviews 

--- 

## Documentation Updates 

If your contribution affects:
- architecture
- APIs 
- system behavior

Please update the relevant documentation under the `docs/` directory.


