# Contributing to Flood-Monitoring-Sentinel2-GEE

Thank you for contributing to this project.

This repository follows a structured, review-based workflow to ensure quality, reproducibility, and academic integrity for the flood monitoring system built using Sentinel-2 satellite imagery, Google Earth Engine, and QGIS.

---

## Contribution Policy

- The `main` branch is **protected**
- Direct commits to `main` are **not allowed**
- All changes must go through a **Pull Request (PR)**
- Every PR requires review and approval before merging

---

## Branching Strategy

Each contributor must work on a **separate branch** created from the `dev` branch.

### Branch Naming Convention

Use clear and descriptive branch names:

- `feat/ndwi-flood-monitoring`
- `feat/gee-export-script`
- `update/cloud-filtering`
- `fix/export-error`
- `docs/readme-update`
- `chore/folder-cleanup`

Create a branch using:
```bash
git checkout -b <branch-name>
```

---

## Workflow

### 1. Clone the repository
```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create a new branch
```bash
git checkout -b <feature-branch>
```

### 3. Make changes and commit
```bash
git add .
git commit -m "<type>: <short description>"
```

### 4. Push your branch
```bash
git push origin <feature-branch>
```

### 5. Open a Pull Request

- **Base branch:** `dev`
- Provide a clear description of:
  - What was implemented
  - Why it was needed
  - Any assumptions or limitations

---

## Pull Request Guidelines

- Keep PRs **small and focused**
- One PR should address **only one feature or fix**
- Avoid unrelated changes in the same PR
- Resolve all review comments before requesting approval

---

## Testing and Validation

Before submitting a PR:

- Ensure Google Earth Engine scripts run without errors
- Verify GeoTIFF outputs are generated correctly
- Confirm no existing functionality is broken
- Follow the current folder structure and naming conventions

---

## Commit Message Guidelines

All commit messages must follow this format:
```
<type>: <short description>
```

### Allowed Commit Types:

- `feat`: Adding a new feature
- `fix`: Fixing a bug or incorrect behavior
- `update`: Improving or modifying existing functionality
- `refactor`: Restructuring code without changing behavior
- `docs`: Documentation-only changes
- `chore`: Maintenance or cleanup tasks
- `test`: Adding or updating tests

### Examples:
```
feat: add NDWI-based flood detection script
update: improve cloud filtering logic
fix: resolve GeoTIFF export error
docs: update methodology section
chore: remove unused scripts
```

---

## Access Control

- Only authorized maintainers can merge PRs into `main`
- Contributors do not have permission to bypass reviews

---

## Documentation Updates

If your contribution affects:

- system workflow
- data processing logic
- output generation
- project architecture

Please update the relevant documentation under the `docs/` directory.
