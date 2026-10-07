/**
 * Site-wide settings that come from the environment at build time.
 * Set them in CI (see .github/workflows/deploy-site.yml) or a local `.env`.
 */

/** Your Kaggle username. When set, lessons link to their published Kaggle notebooks. */
export const KAGGLE_USERNAME: string = import.meta.env.PUBLIC_KAGGLE_USERNAME ?? '';

/** e.g. https://github.com/you/pytorch-for-humans. Used for "view source" links. */
export const REPO_URL: string = import.meta.env.PUBLIC_REPO_URL ?? '';

/** The placeholder used in each lesson's kernel-metadata.json `id`. */
export const KAGGLE_USERNAME_PLACEHOLDER = '{KAGGLE_USERNAME}';
