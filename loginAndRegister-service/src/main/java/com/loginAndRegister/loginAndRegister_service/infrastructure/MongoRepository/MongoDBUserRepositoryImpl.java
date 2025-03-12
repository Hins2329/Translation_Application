package com.loginAndRegister.loginAndRegister_service.infrastructure.MongoRepository;

import com.loginAndRegister.loginAndRegister_service.Domain.model.User;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.mongodb.core.MongoTemplate;

public class MongoDBUserRepositoryImpl implements MongoDBUserRepository {
    @Autowired
    private MongoTemplate mongoTemplate;

    private static final String COLLECTION_NAME = "users";  // Collection name in MongoDB

    @Override
    public void add(User user) {
        mongoTemplate.save(user, COLLECTION_NAME);  // Save user to the "users" collection
    }

    @Override
    public void delete(String username) {
        User user = mongoTemplate.findById(username, User.class, COLLECTION_NAME);
        if (user != null) {
            mongoTemplate.remove(user, COLLECTION_NAME);  // Remove user from the "users" collection
        }
    }

    @Override
    public String checking(String username) {
        User user = mongoTemplate.findById(username, User.class, COLLECTION_NAME);
        if (user != null) {
            return user.getPassword();  // Return password if user exists
        }
        return null;  // Return null if user doesn't exist
    }




//    @Autowired
//    private MongoTemplate mongoTemplate;
//
//    private final String COLLECTION_NAME = "users";  // MongoDB collection name
//
//    @Override
//    public void add(User user) {
//        // Insert or update the user in the collection
//        mongoTemplate.save(user, COLLECTION_NAME);
//    }
//
//    @Override
//    public void delete(User user) {
//        mongoTemplate.remove(user, COLLECTION_NAME);  // Remove the user from the collection
//    }
//
//    @Override
//    public String checking(String username) {
//        return "";
//    }
}
