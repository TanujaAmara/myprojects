const express = require("express");
const path = require("path");

const app = express();
const PORT = 3000;

// Set EJS as the template engine
app.set("view engine", "ejs");

// Middleware for static files
app.use(express.static(path.join(__dirname, "public")));

// Routes
app.get("/", (req, res) => {
  res.render("index", { name: "Tanuja" });
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
