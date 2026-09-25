# Runbook: Database Failure Response

## Overview

**Severity:** P0 - Critical  
**Estimated Time:** 30-60 minutes  
**Prerequisites:** Database admin access, AWS console access

## Symptoms

- Database connection errors
- High error rates in API logs
- Database health check failures
- Alerts: `PostgreSQLDown`, `DatabaseConnectionFailed`

## Initial Assessment (5 minutes)

### 1. Verify the Issue

```bash
# Check database connectivity
psql -h $DB_HOST -U $DB_USER -d wavefield -c "SELECT 1;"

# Check database status in Kubernetes
kubectl get pods -n wavefield-llm -l app=postgresql

# Check RDS status (if using AWS RDS)
aws rds describe-db-instances --db-instance-identifier wavefield-prod
```

### 2. Check Monitoring

- Grafana: Database dashboard
- CloudWatch: RDS metrics
- Prometheus: `pg_up` metric
- Application logs: Connection errors

### 3. Determine Scope

- [ ] Single database instance down
- [ ] Primary database down, replica available
- [ ] All database instances down
- [ ] Database corrupted
- [ ] Network connectivity issue

## Response Procedures

### Scenario A: Single Instance Down (RTO: 15 min)

#### Step 1: Verify Replica Health
```bash
# Check replica status
kubectl get pods -n wavefield-llm -l app=postgresql,role=replica

# Test replica connectivity
psql -h $REPLICA_HOST -U $DB_USER -d wavefield -c "SELECT 1;"
```

#### Step 2: Redirect Traffic to Replica
```bash
# Update service to point to replica
kubectl patch service postgresql -n wavefield-llm \
  -p '{"spec":{"selector":{"role":"replica"}}}'

# Verify traffic redirection
kubectl get endpoints postgresql -n wavefield-llm
```

#### Step 3: Investigate Primary Failure
```bash
# Check pod logs
kubectl logs -n wavefield-llm postgresql-0 --tail=100

# Check events
kubectl describe pod -n wavefield-llm postgresql-0

# Check disk space
kubectl exec -n wavefield-llm postgresql-0 -- df -h
```

#### Step 4: Restart or Replace Primary
```bash
# If recoverable, restart
kubectl delete pod -n wavefield-llm postgresql-0

# If not recoverable, restore from backup
# See "Restore from Backup" section below
```

### Scenario B: Primary Down, Promote Replica (RTO: 20 min)

#### Step 1: Promote Replica to Primary
```bash
# Connect to replica
kubectl exec -it -n wavefield-llm postgresql-replica-0 -- bash

# Promote to primary
pg_ctl promote -D /var/lib/postgresql/data

# Verify promotion
psql -U postgres -c "SELECT pg_is_in_recovery();"
# Should return 'f' (false)
```

#### Step 2: Update Application Configuration
```bash
# Update connection string
kubectl set env deployment/wavefield-api \
  -n wavefield-llm \
  DATABASE_HOST=postgresql-replica-0.postgresql-replica

# Restart application pods
kubectl rollout restart deployment/wavefield-api -n wavefield-llm
```

#### Step 3: Verify Application Connectivity
```bash
# Check application logs
kubectl logs -n wavefield-llm -l app=wavefield-api --tail=50

# Test API endpoint
curl https://api.wavefield-llm.example.com/health
```

#### Step 4: Rebuild Failed Primary as New Replica
```bash
# Create new replica from promoted primary
# Follow standard replica setup procedure
```

### Scenario C: Complete Database Failure (RTO: 30-60 min)

#### Step 1: Declare Incident
```bash
# Notify team
./scripts/notify-incident.sh "P0: Complete database failure"

# Update status page
./scripts/update-status.sh "Database maintenance in progress"
```

#### Step 2: Provision New Database Instance
```bash
# Using Terraform
cd deployment/cloud/aws/terraform
terraform apply -target=aws_db_instance.wavefield

# Or using kubectl
kubectl apply -f deployment/kubernetes/postgresql-statefulset.yaml
```

#### Step 3: Restore from Backup

##### Option A: Restore from Latest Backup
```bash
# Download latest backup
aws s3 cp s3://wavefield-backups/postgres/latest.dump /tmp/

# Restore database
pg_restore -h $NEW_DB_HOST -U $DB_USER -d wavefield \
  -c -v /tmp/latest.dump

# Verify restoration
psql -h $NEW_DB_HOST -U $DB_USER -d wavefield \
  -c "SELECT COUNT(*) FROM users;"
```

##### Option B: Point-in-Time Recovery
```bash
# Restore base backup
pg_restore -h $NEW_DB_HOST -U $DB_USER -d wavefield \
  /backups/base_backup.dump

# Configure recovery
cat > /var/lib/postgresql/data/recovery.conf << EOF
restore_command = 'aws s3 cp s3://wavefield-wal-archive/%f %p'
recovery_target_time = '$(date -u -d '5 minutes ago' '+%Y-%m-%d %H:%M:%S')'
EOF

# Start PostgreSQL
pg_ctl start -D /var/lib/postgresql/data
```

#### Step 4: Verify Data Integrity
```bash
# Run integrity checks
psql -h $NEW_DB_HOST -U $DB_USER -d wavefield << EOF
-- Check table counts
SELECT 
  schemaname,
  tablename,
  n_live_tup as row_count
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;

-- Check for corruption
SELECT * FROM pg_stat_database WHERE datname = 'wavefield';

-- Verify critical tables
SELECT COUNT(*) FROM users;
SELECT COUNT(*) FROM requests;
SELECT COUNT(*) FROM models;
EOF
```

#### Step 5: Update Application Configuration
```bash
# Update database connection
kubectl create secret generic wavefield-db-secret \
  --from-literal=host=$NEW_DB_HOST \
  --from-literal=password=$NEW_DB_PASSWORD \
  -n wavefield-llm \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart applications
kubectl rollout restart deployment/wavefield-api -n wavefield-llm
kubectl rollout restart deployment/wavefield-inference -n wavefield-llm
```

#### Step 6: Verify System Health
```bash
# Run smoke tests
./deployment/ci-cd/scripts/smoke-test.sh api.wavefield-llm.example.com

# Check error rates
curl -s "http://prometheus:9090/api/v1/query?query=rate(wavefield_api_errors_total[5m])"

# Monitor for 15 minutes
watch -n 30 'kubectl get pods -n wavefield-llm'
```

### Scenario D: Database Corruption (RTO: 45-90 min)

#### Step 1: Assess Corruption Extent
```bash
# Check for corruption
psql -h $DB_HOST -U $DB_USER -d wavefield << EOF
-- Check database integrity
SELECT datname, pg_database_size(datname) 
FROM pg_database 
WHERE datname = 'wavefield';

-- Check for corrupted indexes
REINDEX DATABASE wavefield;

-- Check table integrity
SELECT * FROM pg_stat_user_tables WHERE schemaname = 'public';
EOF
```

#### Step 2: Attempt Repair (if minor corruption)
```bash
# Vacuum and analyze
psql -h $DB_HOST -U $DB_USER -d wavefield << EOF
VACUUM FULL ANALYZE;
REINDEX DATABASE wavefield;
EOF
```

#### Step 3: Restore from Backup (if major corruption)
```bash
# Stop application writes
kubectl scale deployment wavefield-api --replicas=0 -n wavefield-llm

# Identify last known good backup
aws s3 ls s3://wavefield-backups/postgres/ | tail -20

# Restore from backup (see Scenario C, Step 3)
```

## Post-Incident Actions

### 1. Verify System Stability (30 minutes)
```bash
# Monitor key metrics
- Error rates < 0.1%
- Response times < 200ms p95
- Database connections stable
- No alerts firing
```

### 2. Document Incident
```bash
# Create incident report
cat > incident-$(date +%Y%m%d).md << EOF
# Database Failure Incident Report

**Date:** $(date)
**Duration:** [START] - [END]
**Impact:** [DESCRIPTION]
**Root Cause:** [ANALYSIS]
**Resolution:** [STEPS TAKEN]
**Lessons Learned:** [IMPROVEMENTS]
EOF
```

### 3. Update Monitoring
- Add alerts for identified gaps
- Update dashboards
- Adjust thresholds if needed

### 4. Schedule Post-Mortem
- Within 48 hours
- All stakeholders invited
- Action items assigned

## Prevention

### Proactive Measures
1. Regular backup testing (weekly)
2. Database health monitoring
3. Capacity planning
4. Automated failover testing
5. Connection pool tuning

### Monitoring Alerts
- Database down
- High connection count
- Slow queries
- Replication lag
- Disk space low

## Rollback Procedures

If restoration fails or causes issues:

```bash
# Revert to previous database
kubectl patch service postgresql -n wavefield-llm \
  -p '{"spec":{"selector":{"version":"previous"}}}'

# Restore application configuration
kubectl rollout undo deployment/wavefield-api -n wavefield-llm
```

## Escalation

### Level 1: On-Call Engineer (0-15 min)
- Initial assessment
- Basic troubleshooting
- Failover to replica

### Level 2: Database Team Lead (15-30 min)
- Complex recovery procedures
- Backup restoration
- Data integrity verification

### Level 3: Infrastructure Director (30+ min)
- Major incidents
- Multi-region failover
- Executive communication

## Contact Information

- **On-Call Engineer:** PagerDuty rotation
- **Database Team:** db-team@wavefield-llm.example.com
- **Infrastructure Lead:** +1-555-INFRA-01
- **Incident Commander:** incidents@wavefield-llm.example.com

## Related Runbooks

- [Service Outage Response](./service-outage.md)
- [Data Corruption Response](./data-corruption.md)
- [Performance Degradation](./performance-degradation.md)

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-01-15 | Infrastructure Team | Initial version |