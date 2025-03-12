package com.translation_service.translation_service.user.infras.persistence;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import java.util.List;

@Document(collection = "users")
public class UserDocument {

    private Long id;
    private String username;
    private String password;
    private String role;
    private List<String> sentences;

    public UserDocument(){};


    public UserDocument(Long id, String username, String password, String role, List<String> sentences) {
        this.id = id;
        this.username = username;
        this.password = password;
        this.role = role;
        this.sentences = sentences;
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

    public List<String> getSentences() {
        return sentences;
    }

    public void setSentences(List<String> sentences) {
        this.sentences = sentences;
    }
}