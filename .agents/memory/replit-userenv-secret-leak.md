---
name: Replit .replit userenv secret leak
description: How plaintext secrets can end up committed in .replit's [userenv.shared] section and how to remediate.
---

Secrets (API tokens, bot tokens, etc.) can sometimes end up written in plaintext
directly into `.replit` under a `[userenv]` / `[userenv.shared]` table instead of
going through the proper Secrets store. This can happen silently (e.g. an
auto-commit) and gets picked up by git history immediately.

**Why:** `.replit` is a plain, git-tracked config file — anything written there
is exposed in the repo/history, unlike real Secrets which are stored outside
version control. `viewEnvVars({ type: "all" })` will show these as plaintext
`envVars.shared` entries (not under `secrets`), which is the tell.

**How to apply:** If you spot credential-looking keys/values in `.replit`'s
`[userenv.shared]` or in `envVars.shared` output:
1. `deleteEnvVars({ environment: "shared", keys: [...] })` to remove the plaintext copies.
2. Re-request the values properly via `requestSecrets` so they land in the real Secrets store.
3. Rewrite `.replit` via `verifyAndReplaceDotReplit` (direct edits are blocked) to drop the empty `[userenv]`/`[userenv.shared]` tables.
4. Because the plaintext value was already committed to git history, treat the credential as compromised and tell the user to rotate/regenerate it at the source (e.g. BotFather for a Telegram bot token) — removing it from the working tree does not scrub history.
