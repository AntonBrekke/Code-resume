from MEK_data_class import *

"""
Calculating circulation in region given by 
"corners" directly, and using Stokes theorem
"""

corners = [(35,160,70,170),(35,85,70,100),(35,50,70,60)]

data_div_curl = div_curl(u, v, dx=0.5, dy=0.5)
data_div_curl.circulation(corners)

# From terminal:
"""
Rectangle 1:
 Direct : 2695.5140926958193
 Stokes : 2621.5586962838092

Each sides :
 C1: 70100.52387861427 (bottom)
 C2: 266.2735761585868 (right)
 C3: -68332.85609978675 (top)
 C4: 661.5727377096991 (left)

Rectangle 2:
 Direct : -60976.60016211555
 Stokes : -61482.54098933286

Each sides :
 C1: 198.47559740489237 (bottom)
 C2: 300.21661027011714 (right)
 C3: -61243.464778495945 (top)
 C4: -231.82759129461652 (left)

Rectangle 3:
 Direct : 9.521016433026077
 Stokes : -12.214333864213021

Each sides :
 C1: 5133.347850903835 (bottom)
 C2: 207.91001043390142 (right)
 C3: -5410.039721925996 (top)
 C4: 78.30287702128548 (left)
"""
