package com.loginAndRegister.loginAndRegister_service.Domain.model;

import org.springframework.context.annotation.Bean;


public class User {
    private String username;
    private String password;
    private String role;
    private Integer id;

    public User(String username, String password, String role, Integer id) {
        this.username = username;
        this.password = password;
        this.role = role;
        this.id = id;
    }

    public User() {
    }

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public String getRole() {
        return role;
    }

    public void setRole(String role) {
        this.role = role;
    }

    @Override
    public String toString() {
        return "User{" +
                "username='" + username + '\'' +
                ", password='" + password + '\'' +
                ", role='" + role + '\'' +
                ", id=" + id +
                '}';
    }
}
