library(tidyverse)
library(ggplot2)
library(plotly)

D = read.csv('results.csv') %>%
  mutate(EGM_index = as.factor(EGM_index))

D %>%
  rename(a1=B1, a2=B2, a3=B3, mw1=MW1, w1=W1, w2=W2, w3=W3, v1=V1, v2=V2) %>%
  
  gather(key=type,value=value, -growth_rate, -EGM_index) %>%
  ggplot(mapping=aes(x = growth_rate, y = value, colour = EGM_index)) +
  facet_wrap(~type, scales = 'free') +
  geom_point(size=0.5) + xlab('Growth rate') + ylab('') + 
  guides(fill=guide_legend(title="EGM index"))

D %>%
  select(-V1, -V2, -e1, -e2, -ribosome_concentration) %>%
  rename(a1=B1, a2=B2, a3=B3, mw1=MW1, w1=W1, w2=W2, w3=W3) %>%
  gather(key=type,value=value, -growth_rate, -EGM_index) %>%
  ggplot(mapping=aes(x = growth_rate, y = value, colour = EGM_index)) +
  facet_wrap(~type, scales = 'free') +
  geom_point(size=0.5) + xlab('Growth rate') + ylab('Concentration / time') + 
  guides(fill=guide_legend(title="EGM index"))

D %>%
  select(-V1, -V2, -e1, -e2, -ribosome_concentration) %>%
  rename(a1=B1, a2=B2, a3=B3, mw1=MW1, w1=W1, w2=W2, w3=W3) %>%
  mutate(a1=a1/0.5, a2=a2/0.8) %>%
  filter(a1 <= 100) %>%
  filter(a2 <= 100) %>%
  select(a1, a2, a3, growth_rate, EGM_index) %>%
  gather(key=type,value=value, -growth_rate, -EGM_index) %>%
  ggplot(mapping=aes(x = growth_rate, y = value, colour = EGM_index)) +
  facet_wrap(~type, scales = 'free') +
  geom_point(size=0.5) + xlab('Growth rate') + ylab('Ribosomal occupation') + 
  guides(fill=guide_legend(title="EGM index"))

D %>%
  plot_ly(x = ~B1, y = ~B2, z = ~B3, symbol = ~EGM_index, color = ~B4, type = 'scatter3d',
        marker = list(symbol = 'line', sizemode = 'diameter'), sizes = c(5, 150))
