// Timeseries chart 
const svg = d3.select("body") 
   .append("svg")
   .attr("width", 500)
   .attr("height", 50);
   
svg.selectAll("rect") 
   .data(data)
   .enter()
   .append("rect")
     .attr("x", (d, i) => i * 10) 
     .attr("y", (d, i) => h - d)
     .attr("width", 10)
     .attr("height", (d, i) => d)
     .attr("fill", "green");

