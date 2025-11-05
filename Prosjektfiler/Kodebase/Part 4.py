# Part 4: Discuss how the choice of k influences the results.
# Valget av k gir utslag på hvor mange grupperinger som dannes. I oppgaven med Iris er det naturlig (for meg ivhertfall)
# at valget faller på 3 grupper, fordi det er 3 hovedgrupper i datasettet.
# Dette kommer og til syne i knekkpunktet på grafen på plot 1 og 2 i Task 3.
# For høy k (oversplitting) vil danne for mange grupper i forhold til hva som er naturlig som ved k = 4 og over,
# hvor en undergruppe blir delt i to. For lav k gjør at to undergrupper blir satt i samme bolk, og det er også
# lite hensiktsmessig.
#
# Silhoutette score kan bistå her. Den øker når punktene passer godt i egen klynge og synker når k er for høy eller lav.
# Et albuepunkt, eller knekkpunkt i inertia kurven samt en topp i silhouette kan indikere en passende k.
