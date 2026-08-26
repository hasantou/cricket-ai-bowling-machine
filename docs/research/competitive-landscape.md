# Competitive landscape — source notes

Working notes on existing bowling-machine products, kept here so claims in
the planning docs are traceable back to a source.

## BOLA Professional

- Line & length adjusted via a large ball joint plus vernier fine
  adjusters; pace controlled in 1 mph increments.
- Swing controlled via push buttons (away swing / inswing), from gentle
  movement to pronounced swing.
- Spin simulated by angling the delivery head with directional controls.
- **Electronic random mode**: the Professional model varies speed and/or
  swing on each ball automatically via the feeder.
- Automatic feeder holds 28 balls, 7 or 11-second selectable intervals;
  key-fob remote on/off.
- Source: [bola.co.uk/cricket.html](https://www.bola.co.uk/cricket.html)

## ProBatter PX3

- Projects a real bowler's run-up and action for realistic visual timing.
- Large programmable library of bowler styles and speeds, set by a coach
  in advance.
- Source: [probatter.com/projects/px3-cricket-simulator](https://probatter.com/projects/px3-cricket-simulator/)

## The gap this project targets

Neither product senses the batter's actual, in-session performance and
adapts the sequence in response — variation is either scheduled/random
(BOLA) or pre-programmed by a coach (ProBatter). See
`docs/product/AI_Adaptive_Bowling_Machine_Program_Plan.docx`, Section 7,
for the full comparison.

## Open research questions (not yet answered)

- Real-world pricing for BOLA Professional and ProBatter PX3, for
  competitive pricing analysis.
- Market sizing: number of academies/clubs in target regions, addressable
  spend on training technology (flagged as a placeholder in the visa
  business plan, Section 4.2).
- Any existing patents in adaptive/reactive bowling-machine control worth
  checking before committing to a specific mechanical approach.
