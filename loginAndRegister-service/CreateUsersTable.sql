CREATE DATABASE my_database;
USE my_database;
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    password VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    UNIQUE(username)  -- Ensure the username is unique
);