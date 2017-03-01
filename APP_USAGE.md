Crossfiltering user interactive workflow:

1. Looks at the periodic table and structure widget

2. Selects the structure widget
   * updates the elements that can be selected get highlighted.

3. Selects the element widget (periodic table)
   * updates the property choices, code, exchange that can
    be selected in the respective widgets

4. Selects the property widget
   * updates the code widget (will remain the same mostly)

5. Selects the code widgets
   * updates the exchange widget

6. Selects the exchange widgets
   * final selection, updates the plottables (mostly fixed)
     - value vs. k-point density
     - value_error vs

     this means
     x = ['k-point density', 'value', 'value_error']

7. Selects the plottables for x-y, in future x-y-z
   - options are values vs. k-point
   - update what statistical tools can be used on the data
   - update what plot types are available too

8. Selects plot type
   * histogram for data in the database of the chosen specs
   * scatter
