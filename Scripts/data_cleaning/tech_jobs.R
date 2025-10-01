library(readabs)
library(dplyr)

tech_occupations_4d <- c(
  '1350 ICT Managers nfd',	 
  '1351 ICT Managers',  
  '2232 ICT Trainers',  
  '2247 Management and Organisation Analysts', 
  '2249 Other Information and Organisation Professionals',	 
  '2252 ICT Sales Professionals',  
  '2324 Graphic and Web Designers, and Illustrators',  
  '2334 Electronics Engineers',  
  '2600 ICT Professionals nfd',  
  '2610 Business and Systems Analysts, and Programmers nfd',  
  '2611 ICT Business and Systems Analysts',  
  '2612 Multimedia Specialists and Web Developers',  
  '2613 Software and Applications Programmers',  
  '2620 Database and Systems Administrators, and ICT Security Specialists nfd',	 
  '2621 Database and Systems Administrators, and ICT Security Specialists',	 
  '2630 ICT Network and Support Professionals nfd',  
  '2631 Computer Network Professionals', 	 
  '2632 ICT Support and Test Engineers',  
  '2633 Telecommunications Engineering Professionals',  
  '3100 Engineering, ICT and Science Technicians nfd',  
  '3124 Electronic Engineering Draftspersons and Technicians',  
  '3130 ICT and Telecommunications Technicians nfd',  
  '3131 ICT Support Technicians',  
  '3132 Telecommunications Technical Specialists',  
  '3424 Telecommunications Trades Workers'
)

labor_force_survey <- read_lfs_datacube("EQ08") 

tech_occupations_lfs <- labor_force_survey |> filter(occupation_of_main_job__anzsco_2013_v1.2 %in% tech_occupations_4d)

# tech industry LFS
#Industry name
#57 Internet Publishing and Broadcasting
#58 Telecommunications Services
#59 Internet Service Providers, Web Search Portals and Data Processing Services
#70 Computer System Design and Related Services
