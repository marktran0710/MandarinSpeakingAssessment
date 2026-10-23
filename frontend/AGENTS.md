# Frontend working rules

Read [FRONTEND_RULES.md](FRONTEND_RULES.md) before changing code under `frontend/`.

- Keep new learner-facing strings in the five-language translation table.
- Call this repository's learning API from React. Keep speaking assessment behind authenticated server routes; do not access either service's database or files from React.
- Keep session credentials in cookies, include CSRF on mutations, and never use a browser-supplied learner ID for ownership.
- Make every new control usable by keyboard and show loading, error and retry states.
- Run `npm run build` and review affected screens at desktop and mobile widths before claiming UI work complete.
