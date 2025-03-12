package com.translation_service.translation_service.common.model;

import java.util.List;

public class User {
    private Long id;
    private String username;
    private String role;
    private List<String> sentences;
    private String password;

    public User() {
    }

    public User(Long id, String username, String role, List<String> sentences, String password) {
        this.id = id;
        this.username = username;
        this.role = role;
        this.sentences = sentences;
        this.password = password;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getRole() {
        return role;
    }

    public void setRole(String role) {
        this.role = role;
    }

    public List<String> getSentences() {
        return sentences;
    }

    public void setSentences(List<String> sentences) {
        this.sentences = sentences;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }
}
