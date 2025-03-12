package com.loginAndRegister.loginAndRegister_service.infrastructure.MongoRepository;

import com.loginAndRegister.loginAndRegister_service.Domain.model.User;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface MongoDBUserRepository {
    void add(User user);
    void delete(String username);
    String checking(String username);  // Returns password or null if user doesn't exist
}
//public interface MongoDBUserRepository extends MongoRepository<User, String> {
//
////public interface MongoDBUserRepository {
//
////public interface UserRepository extends MongoRepository<User, String> {
//    User findByUsername(String username);  // Automatically query by username
//
//    void add(User user);
//    void delete(User user);
//    String checking(String username);
//
//}
