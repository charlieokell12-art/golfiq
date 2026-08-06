# GolfIQ

GolfIQ is a golf practice and course-intelligence platform combining a Shopify storefront with a browser-focused computer-vision pipeline.

## Project areas

- `shopify-theme/` — storefront and customer experience
- `vision-engine/` — ball detection, tracking, direction, carry estimation, confidence and coaching
- `dataset-builder/` — tools for importing videos, extracting frames and reviewing labels
- `datasets/` — manifests and dataset metadata only; raw user videos and large binaries are not committed
- `models/` — model manifests, evaluation reports and browser exports
- `calibration/` — camera and measured-shot calibration
- `tests/` — automated validation and release gates
- `docs/` — architecture, data policy and operating instructions

## Accuracy policy

GolfIQ must not report distance or direction as production-ready unless the relevant model passes held-out evaluation on independent phone recordings and measured carry data. Low-confidence results must be withheld rather than invented.

## Required evidence before release

- Independent train, validation and test source groups
- Positive and negative real-phone footage
- Device and lighting diversity
- Measured carry ground truth
- Direction ground truth against an aligned target line
- Browser performance tests on supported iPhone and Android devices
- Published precision, recall, false-track and calibration metrics

## Development milestones

1. Dataset Builder
2. Setup-ball detector
3. Early-flight detector
4. Impact and temporal tracker
5. Direction estimator
6. Carry calibration
7. Swing analysis without a ball
8. Coaching logic
9. Shopify integration and staged release

## Data rights

Only use footage that is owned, licensed for machine-learning use, public-domain, or submitted with explicit permission. Do not scrape arbitrary copyrighted videos or personal footage.
