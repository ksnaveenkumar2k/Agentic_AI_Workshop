from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Create a PDF document
pdf_file = "sample_data.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=letter)
styles = getSampleStyleSheet()
content = []

# Add title
content.append(Paragraph("Extensive Knowledge Base", styles['Heading1']))
content.append(Spacer(1, 12))

# History Section
content.append(Paragraph("History", styles['Heading2']))
content.append(Paragraph("The French Revolution began in 1789, marking a significant shift in European politics. It led to the overthrow of the monarchy and the rise of Napoleon Bonaparte by 1799.", styles['Normal']))
content.append(Paragraph("World War II, starting in 1939, involved major global powers and ended in 1945 with the defeat of the Axis powers, reshaping international boundaries.", styles['Normal']))
content.append(Paragraph("The Renaissance, spanning the 14th to 17th centuries, was a cultural movement that revived interest in art, science, and humanism in Europe.", styles['Normal']))
content.append(Spacer(1, 12))

# Science Section
content.append(Paragraph("Science", styles['Heading2']))
content.append(Paragraph("The Earth orbits the Sun every 365.25 days, which is why we have a leap year. This orbit defines our calendar year.", styles['Normal']))
content.append(Paragraph("Quantum mechanics, developed in the early 20th century, revolutionized our understanding of atomic and subatomic particles.", styles['Normal']))
content.append(Paragraph("The theory of relativity, proposed by Einstein, describes the effects of gravity and motion at high speeds.", styles['Normal']))
content.append(Spacer(1, 12))

# Mathematics Section
content.append(Paragraph("Mathematics", styles['Heading2']))
content.append(Paragraph("Pythagoras' theorem states that in a right-angled triangle, \(a^2 + b^2 = c^2\), where \(c\) is the hypotenuse. This is fundamental in geometry.", styles['Normal']))
content.append(Paragraph("The Fibonacci sequence, starting with 0 and 1, generates subsequent numbers by adding the two preceding ones, e.g., 0, 1, 1, 2, 3, 5.", styles['Normal']))
content.append(Paragraph("Calculus, developed by Newton and Leibniz, provides tools to analyze change and motion through derivatives and integrals.", styles['Normal']))
content.append(Spacer(1, 12))

# Geography Section
content.append(Paragraph("Geography", styles['Heading2']))
content.append(Paragraph("Mount Everest, standing at 8,848 meters, is the highest peak in the world, located in the Himalayas.", styles['Normal']))
content.append(Paragraph("The Amazon Rainforest, spanning multiple South American countries, is known as the 'lungs of the Earth' due to its oxygen production.", styles['Normal']))
content.append(Paragraph("The Sahara Desert, the largest hot desert, covers much of North Africa and influences regional climate patterns.", styles['Normal']))
content.append(Spacer(1, 12))

# Technology Section
content.append(Paragraph("Technology", styles['Heading2']))
content.append(Paragraph("The invention of the internet in the late 20th century transformed global communication and information access.", styles['Normal']))
content.append(Paragraph("Artificial Intelligence, advancing rapidly since the 2000s, powers applications like virtual assistants and autonomous vehicles.", styles['Normal']))
content.append(Paragraph("Blockchain technology, introduced with Bitcoin in 2009, enables secure, decentralized transaction records.", styles['Normal']))
content.append(Spacer(1, 12))

# Literature Section
content.append(Paragraph("Literature", styles['Heading2']))
content.append(Paragraph("William Shakespeare's 'Hamlet,' written around 1600, is a tragedy exploring themes of revenge and madness.", styles['Normal']))
content.append(Paragraph("Jane Austen's 'Pride and Prejudice,' published in 1813, is a classic novel of romance and social commentary.", styles['Normal']))
content.append(Paragraph("George Orwell's '1984,' released in 1949, is a dystopian novel warning about totalitarian surveillance.", styles['Normal']))
content.append(Spacer(1, 12))

# Biology Section
content.append(Paragraph("Biology", styles['Heading2']))
content.append(Paragraph("DNA, discovered by Watson and Crick in 1953, is the molecule that carries genetic information in living organisms.", styles['Normal']))
content.append(Paragraph("Photosynthesis, performed by plants, converts light energy into chemical energy, producing oxygen as a byproduct.", styles['Normal']))
content.append(Paragraph("The human body contains approximately 37.2 trillion cells, each with specialized functions.", styles['Normal']))
content.append(Spacer(1, 12))

# Astronomy Section
content.append(Paragraph("Astronomy", styles['Heading2']))
content.append(Paragraph("The Milky Way galaxy, our home galaxy, contains an estimated 100-400 billion stars and a supermassive black hole at its center.", styles['Normal']))
content.append(Paragraph("The Apollo 11 mission in 1969 marked the first human landing on the Moon, led by Neil Armstrong.", styles['Normal']))
content.append(Paragraph("A light-year, the distance light travels in one year, is about 9.46 trillion kilometers.", styles['Normal']))
content.append(Spacer(1, 12))

# Economics Section
content.append(Paragraph("Economics", styles['Heading2']))
content.append(Paragraph("The Great Depression, beginning in 1929, was a severe global economic downturn lasting through the 1930s.", styles['Normal']))
content.append(Paragraph("Supply and demand principles govern market prices, where scarcity increases value.", styles['Normal']))
content.append(Paragraph("The GDP of a country measures the total monetary value of goods and services produced annually.", styles['Normal']))
content.append(Spacer(1, 12))

# Medicine Section
content.append(Paragraph("Medicine", styles['Heading2']))
content.append(Paragraph("Penicillin, discovered by Alexander Fleming in 1928, was the first antibiotic, revolutionizing medical treatment.", styles['Normal']))
content.append(Paragraph("Vaccines work by stimulating the immune system to recognize and combat specific pathogens.", styles['Normal']))
content.append(Paragraph("The human heart beats approximately 60-100 times per minute at rest, pumping blood throughout the body.", styles['Normal']))
content.append(Spacer(1, 12))

# Art Section
content.append(Paragraph("Art", styles['Heading2']))
content.append(Paragraph("The Mona Lisa, painted by Leonardo da Vinci in the early 16th century, is renowned for its enigmatic expression.", styles['Normal']))
content.append(Paragraph("Impressionism, emerging in the 19th century, focused on light and everyday subjects, led by artists like Monet.", styles['Normal']))
content.append(Paragraph("The Sistine Chapel ceiling, painted by Michelangelo, is a masterpiece of Renaissance art completed in 1512.", styles['Normal']))
content.append(Spacer(1, 12))

# Environmental Science Section
content.append(Paragraph("Environmental Science", styles['Heading2']))
content.append(Paragraph("Climate change, driven by greenhouse gas emissions, is raising global temperatures at an alarming rate.", styles['Normal']))
content.append(Paragraph("Renewable energy sources, such as solar and wind, are critical for reducing carbon footprints.", styles['Normal']))
content.append(Paragraph("Deforestation in the Amazon has led to a loss of biodiversity and increased CO2 levels.", styles['Normal']))

# Build the PDF
doc.build(content)

print(f"Sample PDF generated successfully at {pdf_file}")