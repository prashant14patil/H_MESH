# Size charts

Each `*.json` file in this folder is offered in the Size recommendation picker
(after a dev-server reload or rebuild). Operators can also upload a chart in the
UI; uploaded charts are kept in that browser only.

```json
{
  "id": "brand-mens",
  "name": "Brand men's tops",
  "sizes": ["S", "M", "L"],
  "bounds": {
    "chest": [88, 96, 104, 112],
    "waist": [76, 84, 92, 100],
    "hip":   [92, 100, 108, 116]
  }
}
```

- `bounds` lists, in cm, the lower bound of each size followed by the upper bound
  of the last size, so each list has one more number than `sizes`, strictly increasing.
- `chest` drives the top size; the larger of `waist` and `hip` drives trousers.
  Give at least one; leave out what the brand does not publish.
- Set `"example": true` only for made-up charts, so the UI labels them as examples.

Validation rules live in `validateChart` (`src/components/sizeRecommendation.js`).
